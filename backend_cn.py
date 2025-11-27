#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/11/5 16:11
# @Author : Administrator
# @File : backend_cn.py
# @Project : youtu-graphrag-main
# @Software: PyCharm
# !/usr/bin/env python3
"""
Simple but Complete Youtu-GraphRAG Backend
Integrates real GraphRAG functionality with a simple interface
"""  # 作用：简洁完整的 Youtu-GraphRAG 后端，整合 GraphRAG 核心功能与简单接口

import os
import re
import sys
import json
import asyncio
import glob
import shutil
from typing import List, Dict, Optional
from datetime import datetime

# Add project root to path  # 添加项目根目录到路径，便于导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# FastAPI imports  # FastAPI 相关导入：API 框架、文件上传、WebSocket 等
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from utils.logger import logger  # 日志工具
import ast  # 用于解析三元组字符串


from nacos import NacosClient, NacosException
import socket

# client = NacosClient(**nacos_config)  # 创建客户端实例
server_addresses = 'localhost:8848'
namespace = 'public'
group_name = 'DEFAULT_GROUP'
client = NacosClient(
    server_addresses=server_addresses,
    namespace=namespace,
)  #

# Import document parser  # 尝试导入文档解析器（用于 PDF/DOCX 等）
try:
    from utils.document_parser import get_parser

    DOCUMENT_PARSER_AVAILABLE = True
except ImportError as e:
    DOCUMENT_PARSER_AVAILABLE = False
    logger.warning(f"Document parser not available: {e}")

# Try to import GraphRAG components  # 尝试导入 GraphRAG 核心组件：构建器、检索器、配置
try:
    from models.constructor import kt_gen as constructor
    from models.retriever import agentic_decomposer as decomposer, enhanced_kt_retriever as retriever
    from config import get_config, ConfigManager

    GRAPHRAG_AVAILABLE = True
    logger.info("✅ GraphRAG components loaded successfully")
except ImportError as e:
    GRAPHRAG_AVAILABLE = False
    logger.error(f"⚠️  GraphRAG components not available: {e}")

app = FastAPI(title="Youtu-GraphRAG Unified Interface", version="1.0.0")  # 创建 FastAPI 应用

# Mount static files (assets directory)  # 挂载静态文件：assets（资源）、frontend（前端 UI）
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# CORS middleware  # 添加 CORS 中间件，支持跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables  # 全局变量：活跃 WebSocket 连接、配置
active_connections: Dict[str, WebSocket] = {}
config = None


# 通过 Header 获取用户名  # 用于多用户隔离：从请求头 "X-Username" 获取用户名
async def get_username(x_username: str = Header(None)):
    if not x_username:
        raise HTTPException(status_code=401, detail="No username provided in header 'X-Username'")
    return x_username


class ConnectionManager:  # WebSocket 连接管理器：处理客户端连接、断开、消息广播
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)


manager = ConnectionManager()  # 实例化连接管理器


# Request/Response models  # Pydantic 模型：定义 API 请求/响应结构
class FileUploadResponse(BaseModel):
    success: bool
    message: str
    dataset_name: Optional[str] = None
    files_count: Optional[int] = None


class GraphConstructionRequest(BaseModel):
    dataset_name: str


class GraphConstructionResponse(BaseModel):
    success: bool
    message: str
    graph_data: Optional[Dict] = None


class QuestionRequest(BaseModel):
    question: str
    dataset_name: str


class QuestionResponse(BaseModel):
    answer: str
    sub_questions: List[Dict]
    retrieved_triples: List[str]
    retrieved_chunks: List[str]
    reasoning_steps: List[Dict]
    visualization_data: Dict


def ensure_demo_schema_exists() -> str:
    """Ensure default demo schema exists and return its path."""  # 确保默认 demo schema 存在，并返回路径
    os.makedirs("schemas", exist_ok=True)
    schema_path = "schemas/demo.json"
    if not os.path.exists(schema_path):
        demo_schema = {
            "Nodes": [
                "person", "location", "organization", "event", "object",
                "concept", "time_period", "creative_work", "biological_entity", "natural_phenomenon"
            ],
            "Relations": [
                "is_a", "part_of", "located_in", "created_by", "used_by", "participates_in",
                "related_to", "belongs_to", "influences", "precedes", "arrives_in", "comparable_to"
            ],
            "Attributes": [
                "name", "date", "size", "type", "description", "status",
                "quantity", "value", "position", "duration", "time"
            ]
        }
        with open(schema_path, 'w') as f:
            json.dump(demo_schema, f, indent=2)
    return schema_path


def get_schema_path_for_dataset(username: str, dataset_name: str) -> str:
    """Return dataset-specific schema if present; otherwise fallback to demo schema."""  # 获取数据集特定 schema 或回退到 demo
    if dataset_name and dataset_name != "demo":
        ds_schema = f"schemas/{username}/{dataset_name}.json"
        if os.path.exists(ds_schema):
            return ds_schema
    return ensure_demo_schema_exists()


async def send_progress_update(client_id: str, stage: str, progress: int, message: str):
    """Send progress update via WebSocket"""  # 通过 WebSocket 发送进度更新（stage: 阶段, progress: 进度百分比）
    await manager.send_message({
        "type": "progress",
        "stage": stage,
        "progress": progress,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }, client_id)


# -------- Encoding detection helpers --------  # 编码检测辅助函数：处理多语言文件（如中文）
def _detect_encoding_from_bytes(data: bytes) -> Optional[str]:
    """Detect encoding using chardet if available; return lower-cased encoding name or None."""
    try:
        import chardet  # type: ignore
        result = chardet.detect(data) or {}
        enc = result.get("encoding")
        if enc:
            return enc.lower()
    except Exception:
        pass
    return None


def decode_bytes_with_detection(data: bytes) -> str:
    """Decode bytes to string with encoding detection and robust fallbacks."""  # 解码字节，支持 UTF-8/GBK 等中文编码
    candidates = []
    detected = _detect_encoding_from_bytes(data)
    if detected:
        candidates.append(detected)
    candidates.extend([
        "utf-8", "utf-8-sig", "gb18030", "gbk", "big5",
        "utf-16", "utf-16le", "utf-16be", "latin-1"
    ])
    # De-duplicate while preserving order
    tried = set()
    for enc in candidates:
        if enc in tried or not enc:
            continue
        tried.add(enc)
        try:
            return data.decode(enc)
        except Exception:
            continue
    # Last resort
    return data.decode("utf-8", errors="replace")


async def clear_cache_files(username: str, dataset_name: str):
    """Clear all cache files for a dataset before graph construction"""  # 清理数据集缓存（FAISS 索引、图文件、分块等），添加用户名路径隔离
    try:
        # Clear FAISS cache files
        faiss_cache_dir = f"retriever/faiss_cache_new/{username}/{dataset_name}"
        if os.path.exists(faiss_cache_dir):
            shutil.rmtree(faiss_cache_dir)
            logger.info(f"Cleared FAISS cache directory: {faiss_cache_dir}")

        # Clear output chunks
        chunk_file = f"output/chunks/{username}/{dataset_name}.txt"
        if os.path.exists(chunk_file):
            os.remove(chunk_file)
            logger.info(f"Cleared chunk file: {chunk_file}")

        # Clear output graphs
        graph_file = f"output/graphs/{username}/{dataset_name}_new.json"
        if os.path.exists(graph_file):
            os.remove(graph_file)
            logger.info(f"Cleared graph file: {graph_file}")

        # Clear any other cache files with dataset name pattern
        cache_patterns = [
            f"output/logs/{username}/{dataset_name}_*.log",
            f"output/chunks/{username}/{dataset_name}_*",
            f"output/graphs/{username}/{dataset_name}_*"
        ]

        for pattern in cache_patterns:
            for file_path in glob.glob(pattern):
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        logger.info(f"Cleared cache file: {file_path}")
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        logger.info(f"Cleared cache directory: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clear {file_path}: {e}")

        logger.info(f"Cache cleanup completed for dataset: {dataset_name} under user: {username}")

    except Exception as e:
        logger.error(f"Error clearing cache files for {dataset_name} under {username}: {e}")
        # Don't raise exception, just log the error


# Serve frontend HTML  # 根路径：服务前端 index.html 或 JSON 状态
@app.get("/")
async def read_root():
    # print("@app.get('/')")
    frontend_path = "frontend/index.html"
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {"message": "Youtu-GraphRAG Unified Interface is running!", "status": "ok"}


@app.get("/api/status")  # API 状态检查
async def get_status():
    # print("@app.get('/api/status')  # API 状态检查")
    return {
        "message": "Youtu-GraphRAG Unified Interface is running!",
        "status": "ok",
        "graphrag_available": GRAPHRAG_AVAILABLE
    }


@app.websocket("/ws/{client_id}")  # WebSocket 端点：实时连接管理
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(client_id)


@app.post("/api/upload", response_model=FileUploadResponse)
async def upload_files(
        files: List[UploadFile] = File(...),
        client_id: str = "default",
        username: str = Depends(get_username)
):
    """Upload files and prepare for graph construction"""  # 文件上传端点：生成数据集，解析文件，保存 corpus.json，路径添加用户名隔离
    try:
        # Generate dataset name based on file count
        if len(files) == 1:
            # Single file: use its name
            main_file = files[0]
            original_name = os.path.splitext(main_file.filename)[0]
            # Clean filename to be filesystem-safe
            dataset_name = "".join(c for c in original_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            dataset_name = dataset_name.replace(' ', '_')
        else:
            # Multiple files: create a descriptive name with date
            from datetime import datetime
            date_str = datetime.now().strftime("%Y%m%d")
            dataset_name = f"{len(files)}files_{date_str}"

        # Add counter if dataset already exists
        base_name = dataset_name
        counter = 1
        user_upload_dir = f"data/uploaded/{username}"
        os.makedirs(user_upload_dir, exist_ok=True)  # 确保用户目录存在
        while os.path.exists(f"{user_upload_dir}/{dataset_name}"):
            dataset_name = f"{base_name}_{counter}"
            counter += 1

        upload_dir = f"{user_upload_dir}/{dataset_name}"
        os.makedirs(upload_dir, exist_ok=True)

        await send_progress_update(client_id, "upload", 10, "Starting file upload...")

        # Process uploaded files
        corpus_data = []
        skipped_files: List[str] = []
        processed_count = 0
        allowed_extensions = {".txt", ".md", ".json", ".pdf", ".docx", ".doc"}

        # Initialize document parser if needed
        doc_parser = None
        if DOCUMENT_PARSER_AVAILABLE:
            doc_parser = get_parser()

        for i, file in enumerate(files):
            file_path = os.path.join(upload_dir, file.filename)
            with open(file_path, "wb") as buffer:
                content_bytes = await file.read()
                buffer.write(content_bytes)

            # Process file content using encoding detection
            filename_lower = (file.filename or "").lower()
            ext = os.path.splitext(filename_lower)[1]
            if ext not in allowed_extensions:
                # Skip unsupported file types to avoid processing binary files as text
                logger.warning(f"Skipping unsupported file type: {file.filename}")
                skipped_files.append(file.filename)
                progress = 10 + (i + 1) * 80 // len(files)
                await send_progress_update(client_id, "upload", progress, f"Skipped unsupported file: {file.filename}")
                continue

            # Handle PDF and DOCX/DOC files with document parser
            if ext in ['.pdf', '.docx', '.doc']:
                if not doc_parser:
                    logger.warning(f"Document parser not available, skipping {file.filename}")
                    skipped_files.append(file.filename)
                    progress = 10 + (i + 1) * 80 // len(files)
                    await send_progress_update(client_id, "upload", progress,
                                               f"Skipped {file.filename} (parser unavailable)")
                    continue

                try:
                    text = doc_parser.parse_file(file_path, ext)
                    if text and text.strip():
                        corpus_data.append({
                            "title": file.filename,
                            "text": text
                        })
                        processed_count += 1
                        await send_progress_update(client_id, "upload", 10 + (i + 1) * 80 // len(files),
                                                   f"Parsed {file.filename}")
                    else:
                        logger.warning(f"No text extracted from {file.filename}")
                        skipped_files.append(file.filename)
                        await send_progress_update(client_id, "upload", 10 + (i + 1) * 80 // len(files),
                                                   f"No text in {file.filename}")
                except Exception as e:
                    logger.error(f"Error parsing {file.filename}: {e}")
                    skipped_files.append(file.filename)
                    await send_progress_update(client_id, "upload", 10 + (i + 1) * 80 // len(files),
                                               f"Failed to parse {file.filename}")
                continue

            # Treat plain text formats explicitly (.txt and .md)
            if filename_lower.endswith(('.txt', '.md')):
                text = decode_bytes_with_detection(content_bytes)
                corpus_data.append({
                    "title": file.filename,
                    "text": text
                })
                processed_count += 1
            elif filename_lower.endswith('.json'):
                try:
                    json_text = decode_bytes_with_detection(content_bytes)
                    data_obj = json.loads(json_text)
                    if isinstance(data_obj, list):
                        corpus_data.extend(data_obj)
                    else:
                        corpus_data.append(data_obj)
                    processed_count += 1
                except Exception:
                    # If JSON parsing fails, treat as text
                    text = decode_bytes_with_detection(content_bytes)
                    corpus_data.append({
                        "title": file.filename,
                        "text": text
                    })

            progress = 10 + (i + 1) * 80 // len(files)
            await send_progress_update(client_id, "upload", progress, f"Processed {file.filename}")

        # Ensure at least one valid file processed
        if processed_count == 0:
            msg = "No supported files were uploaded. Allowed: .txt, .md, .json, .pdf, .docx, .doc"
            if skipped_files:
                msg += f"; skipped: {', '.join(skipped_files)}"
            await send_progress_update(client_id, "upload", 0, msg)
            raise HTTPException(status_code=400, detail=msg)

        # Save corpus data
        corpus_path = f"{upload_dir}/corpus.json"
        with open(corpus_path, 'w', encoding='utf-8') as f:
            json.dump(corpus_data, f, ensure_ascii=False, indent=2)

        # Create dataset configuration
        await create_dataset_config()

        await send_progress_update(client_id, "upload", 100, "Upload completed successfully!")

        msg_ok = "Files uploaded successfully"
        if skipped_files:
            msg_ok += f"; skipped unsupported: {', '.join(skipped_files)}"
        return FileUploadResponse(
            success=True,
            message=msg_ok,
            dataset_name=dataset_name,
            files_count=processed_count
        )

    except Exception as e:
        await send_progress_update(client_id, "upload", 0, f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def create_dataset_config():
    """Create dataset configuration"""  # 创建数据集配置，确保 demo schema 存在
    # Ensure default demo schema exists
    ensure_demo_schema_exists()


@app.post("/api/construct-graph", response_model=GraphConstructionResponse)
async def construct_graph(
        request: GraphConstructionRequest,
        client_id: str = "default",
        username: str = Depends(get_username)
):
    """Construct knowledge graph from uploaded data"""  # 构建图端点：清理缓存，初始化 KTBuilder，构建知识图谱，路径添加用户名隔离
    try:
        if not GRAPHRAG_AVAILABLE:
            raise HTTPException(status_code=503,
                                detail="GraphRAG components not available. Please install or configure them.")
        dataset_name = request.dataset_name

        await send_progress_update(client_id, "construction", 2, "Cleaning old cache files...")

        # Clear all cache files before construction
        await clear_cache_files(username, dataset_name)

        await send_progress_update(client_id, "construction", 5, "Initializing graph builder...")

        # Get dataset paths
        corpus_path = f"data/uploaded/{username}/{dataset_name}/corpus.json"
        # Choose schema: dataset-specific or default demo
        schema_path = get_schema_path_for_dataset(username, dataset_name)

        if not os.path.exists(corpus_path):
            # Try demo dataset
            corpus_path = "data/demo/demo_corpus.json"

        if not os.path.exists(corpus_path):
            raise HTTPException(status_code=404, detail="Dataset not found")

        await send_progress_update(client_id, "construction", 10, "Loading configuration and corpus...")

        # Initialize config
        global config
        if config is None:
            config = get_config("config/base_config.yaml")

        # Initialize KTBuilder
        builder = constructor.KTBuilder(
            dataset_name,
            schema_path,
            mode=config.construction.mode,
            config=config
        )

        await send_progress_update(client_id, "construction", 20, "Starting entity-relation extraction...")

        # Build knowledge graph
        def build_graph_sync():
            return builder.build_knowledge_graph(corpus_path)

        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()

        # Run graph construction without simulated progress updates
        knowledge_graph = await loop.run_in_executor(None, build_graph_sync)

        await send_progress_update(client_id, "construction", 95, "Preparing visualization data...")
        # Load constructed graph for visualization
        graph_path = f"output/graphs/{username}/{dataset_name}_new.json"
        graph_vis_data = await prepare_graph_visualization(graph_path)

        await send_progress_update(client_id, "construction", 100, "Graph construction completed!")
        # Notify completion via WebSocket
        try:
            await manager.send_message({
                "type": "complete",
                "stage": "construction",
                "message": "Graph construction completed!",
                "timestamp": datetime.now().isoformat()
            }, client_id)
        except Exception as _e:
            logger.warning(f"Failed to send completion message: {_e}")

        return GraphConstructionResponse(
            success=True,
            message="Knowledge graph constructed successfully",
            graph_data=graph_vis_data
        )

    except Exception as e:
        await send_progress_update(client_id, "construction", 0, f"Construction failed: {str(e)}")
        try:
            await manager.send_message({
                "type": "error",
                "stage": "construction",
                "message": f"Construction failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }, client_id)
        except Exception as _e:
            logger.warning(f"Failed to send error message: {_e}")
        raise HTTPException(status_code=500, detail=str(e))


async def prepare_graph_visualization(graph_path: str) -> Dict:
    """Prepare graph data for visualization"""  # 准备图可视化数据：转换格式为 ECharts 兼容
    try:
        if os.path.exists(graph_path):
            with open(graph_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
        else:
            return {"nodes": [], "links": [], "categories": [], "stats": {}}

        # Handle different graph data formats
        if isinstance(graph_data, list):
            # GraphRAG format: list of relationships
            return convert_graphrag_format(graph_data)
        elif isinstance(graph_data, dict) and "nodes" in graph_data:
            # Standard format: {nodes: [], edges: []}
            return convert_standard_format(graph_data)
        else:
            return {"nodes": [], "links": [], "categories": [], "stats": {}}

    except Exception as e:
        logger.error(f"Error preparing visualization: {e}")
        return {"nodes": [], "links": [], "categories": [], "stats": {}}


def convert_graphrag_format(graph_data: List) -> Dict:
    """Convert GraphRAG relationship list to ECharts format"""  # 转换 GraphRAG 格式为 ECharts
    nodes_dict = {}
    links = []

    # Extract nodes and relationships from the list
    for item in graph_data:
        if not isinstance(item, dict):
            continue

        start_node = item.get("start_node", {})
        end_node = item.get("end_node", {})
        relation = item.get("relation", "related_to")

        # Process start node
        start_id = ""
        end_id = ""
        if start_node:
            start_id = start_node.get("properties", {}).get("name", "")
            if start_id and start_id not in nodes_dict:
                nodes_dict[start_id] = {
                    "id": start_id,
                    "name": start_id[:30],
                    "category": start_node.get("properties", {}).get("schema_type", start_node.get("label", "entity")),
                    "symbolSize": 25,
                    "properties": start_node.get("properties", {})
                }

        # Process end node
        if end_node:
            end_id = end_node.get("properties", {}).get("name", "")
            if end_id and end_id not in nodes_dict:
                nodes_dict[end_id] = {
                    "id": end_id,
                    "name": end_id[:30],
                    "category": end_node.get("properties", {}).get("schema_type", end_node.get("label", "entity")),
                    "symbolSize": 25,
                    "properties": end_node.get("properties", {})
                }

        # Add relationship
        if start_id and end_id:
            links.append({
                "source": start_id,
                "target": end_id,
                "name": relation,
                "value": 1
            })

    # Create categories
    categories_set = set()
    for node in nodes_dict.values():
        categories_set.add(node["category"])

    categories = []
    for i, cat_name in enumerate(categories_set):
        categories.append({
            "name": cat_name,
            "itemStyle": {
                "color": f"hsl({i * 360 / len(categories_set)}, 70%, 60%)"
            }
        })

    nodes = list(nodes_dict.values())

    return {
        "nodes": nodes[:500],  # Limit for better visual effects
        "links": links[:1000],
        "categories": categories,
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(links),
            "displayed_nodes": len(nodes[:500]),
            "displayed_edges": len(links[:1000])
        }
    }


def convert_standard_format(graph_data: Dict) -> Dict:
    """Convert standard {nodes: [], edges: []} format to ECharts format"""  # 转换标准格式为 ECharts
    nodes = []
    links = []
    categories = []

    # Extract unique categories
    node_types = set()
    for node in graph_data.get("nodes", []):
        node_type = node.get("type", "entity")
        node_types.add(node_type)

    for i, node_type in enumerate(node_types):
        categories.append({
            "name": node_type,
            "itemStyle": {
                "color": f"hsl({i * 360 / len(node_types)}, 70%, 60%)"
            }
        })

    # Process nodes
    for node in graph_data.get("nodes", []):
        nodes.append({
            "id": node.get("id", ""),
            "name": node.get("name", node.get("id", ""))[:30],
            "category": node.get("type", "entity"),
            "value": len(node.get("attributes", [])),
            "symbolSize": min(max(len(node.get("attributes", [])) * 3 + 15, 15), 40),
            "attributes": node.get("attributes", [])
        })

    # Process edges
    for edge in graph_data.get("edges", []):
        links.append({
            "source": edge.get("source", ""),
            "target": edge.get("target", ""),
            "name": edge.get("relation", "related_to"),
            "value": edge.get("weight", 1)
        })

    return {
        "nodes": nodes[:500],  # Limit for performance
        "links": links[:1000],
        "categories": categories,
        "stats": {
            "total_nodes": len(graph_data.get("nodes", [])),
            "total_edges": len(graph_data.get("edges", [])),
            "displayed_nodes": len(nodes[:500]),
            "displayed_edges": len(links[:1000])
        }
    }


@app.post("/api/ask-question", response_model=QuestionResponse)
async def ask_question(
        request: QuestionRequest,
        client_id: str = "default",
        username: str = Depends(get_username)
):
    """Process question using agent mode (iterative retrieval + reasoning) and return answer."""  # 查询端点：问题分解、检索、迭代推理，路径添加用户名隔离
    try:
        if not GRAPHRAG_AVAILABLE:
            raise HTTPException(status_code=503,
                                detail="GraphRAG components not available. Please install or configure them.")
        dataset_name = request.dataset_name
        question = request.question

        await send_progress_update(client_id, "retrieval", 10, "Initializing retrieval system (agent mode)...")

        graph_path = f"output/graphs/{username}/{dataset_name}_new.json"
        schema_path = get_schema_path_for_dataset(username, dataset_name)
        if not os.path.exists(graph_path):
            graph_path = "output/graphs/demo_new.json"
        if not os.path.exists(graph_path):
            raise HTTPException(status_code=404, detail="Graph not found. Please construct graph first.")

        # Config & components
        global config
        if config is None:
            config = get_config("config/base_config.yaml")

        graphq = decomposer.GraphQ(dataset_name, config=config)
        kt_retriever = retriever.KTRetriever(
            dataset_name,
            graph_path,
            recall_paths=config.retrieval.recall_paths,
            schema_path=schema_path,
            top_k=config.retrieval.top_k_filter,
            mode="agent",  # force agent mode
            config=config
        )

        await send_progress_update(client_id, "retrieval", 40, "Building indices...")
        loop = asyncio.get_running_loop()
        # Offload index building to thread executor to avoid blocking event loop
        await loop.run_in_executor(None, kt_retriever.build_indices)

        # Notify QA start via WS so frontend can show immediate progress
        try:
            await manager.send_message({
                "type": "qa_update",
                "stage": "start",
                "message": "Question processing started",
                "dataset": dataset_name,
                "question": question,
                "timestamp": datetime.now().isoformat()
            }, client_id)
            await asyncio.sleep(0)
        except Exception as _e:
            logger.debug(f"QA start ws send failed: {_e}")

        # Helper functions (reuse a simplified version of main.py logic)
        def _dedup(items):
            return list({x: None for x in items}.keys())

        def _merge_chunk_contents(ids, mapping):
            return [mapping.get(i, f"[Missing content for chunk {i}]") for i in ids]

        # Step 1: decomposition
        await send_progress_update(client_id, "retrieval", 50, "Decomposing question...")
        try:
            # Offload decomposition to executor
            loop = asyncio.get_running_loop()
            decomposition = await loop.run_in_executor(None, lambda: graphq.decompose(question, schema_path))
            sub_questions = decomposition.get("sub_questions", [])
            involved_types = decomposition.get("involved_types", {})
            try:
                await manager.send_message({
                    "type": "qa_update",
                    "stage": "decompose",
                    "sub_questions_count": len(sub_questions),
                    "sub_questions": [sq.get("sub-question", "") for sq in sub_questions][:5],
                    "timestamp": datetime.now().isoformat()
                }, client_id)
                await asyncio.sleep(0.05)
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Decompose failed: {e}")
            sub_questions = [{"sub-question": question}]
            involved_types = {"nodes": [], "relations": [], "attributes": []}
            decomposition = {"sub_questions": sub_questions, "involved_types": involved_types}

        reasoning_steps = []
        all_triples = set()
        all_chunk_ids = set()
        all_chunk_contents: Dict[str, str] = {}

        # Step 2: initial retrieval for each sub-question
        await send_progress_update(client_id, "retrieval", 65, "Initial retrieval...")
        import time as _time
        for idx, sq in enumerate(sub_questions):
            sq_text = sq.get("sub-question", question)
            start_t = _time.time()

            # Offload retrieval to thread executor to avoid blocking event loop
            def _run_retrieval():
                return kt_retriever.process_retrieval_results(
                    sq_text,
                    top_k=config.retrieval.top_k_filter,
                    involved_types=involved_types
                )

            retrieval_results, elapsed = await loop.run_in_executor(None, _run_retrieval)
            triples = retrieval_results.get('triples', []) or []
            chunk_ids = retrieval_results.get('chunk_ids', []) or []
            chunk_contents = retrieval_results.get('chunk_contents', []) or []
            if isinstance(chunk_contents, dict):
                for cid, ctext in chunk_contents.items():
                    all_chunk_contents[cid] = ctext
            else:
                for i_c, cid in enumerate(chunk_ids):
                    if i_c < len(chunk_contents):
                        all_chunk_contents[cid] = chunk_contents[i_c]
            all_triples.update(triples)
            all_chunk_ids.update(chunk_ids)
            reasoning_steps.append({
                "type": "sub_question",
                "question": sq_text,
                "triples": triples[:10],
                "triples_count": len(triples),
                "chunks_count": len(chunk_ids),
                "processing_time": elapsed,
                "chunk_contents": list(all_chunk_contents.values())[:3]
            })

            # Stream this sub-question's partial result to frontend via WebSocket
            try:
                await manager.send_message({
                    "type": "qa_update",
                    "stage": "sub_question",
                    "index": idx + 1,
                    "total": len(sub_questions),
                    "question": sq_text,
                    "triples_preview": list(dict.fromkeys(triples))[:5],
                    "triples_count": len(triples),
                    "chunks_count": len(chunk_ids),
                    "processing_time": elapsed,
                    "timestamp": datetime.now().isoformat()
                }, client_id)
                # yield to event loop to flush WS frames
                await asyncio.sleep(0)
            except Exception as _e:
                logger.debug(f"QA update ws send failed for sub_question {idx + 1}: {_e}")

        # Step 3: IRCoT iterative refinement
        await send_progress_update(client_id, "retrieval", 75, "Iterative reasoning...")
        try:
            await manager.send_message({
                "type": "qa_update",
                "stage": "ircot_start",
                "message": "Starting iterative reasoning",
                "timestamp": datetime.now().isoformat()
            }, client_id)
            await asyncio.sleep(0.05)
        except Exception:
            pass
        max_steps = getattr(getattr(config.retrieval, 'agent', object()), 'max_steps', 3)
        current_query = question
        thoughts = []

        # Initial answer attempt
        initial_triples = _dedup(list(all_triples))
        initial_chunk_ids = list(set(all_chunk_ids))
        initial_chunk_contents = _merge_chunk_contents(initial_chunk_ids, all_chunk_contents)
        context_initial = "=== Triples ===\n" + "\n".join(initial_triples[:20]) + "\n=== Chunks ===\n" + "\n".join(
            initial_chunk_contents[:10])
        init_prompt = kt_retriever.generate_prompt(question, context_initial)

        # ========== 新增代码保存提示词开始 (New Code Starts) ==========

        # 1. 定义一个专门存放日志的文件夹名
        log_dir = "prompt_logs"

        # 2. 检查这个文件夹是否存在，如果不存在，则创建它
        #    (os 库已在文件顶部导入)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 3. 创建一个基于当前时间的唯一文件名 (例如: initial_prompt_20251110_113000.txt)
        #    我们将其命名为 "initial_prompt" 因为这是 IRCoT 流程的第一个提示词
        #    (datetime 类已在文件顶部从 datetime 模块导入)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(log_dir, f"initial_prompt_{timestamp}.txt")

        try:
            # 4. 使用 'w' (写入) 模式和 'utf-8' 编码打开文件
            with open(filename, "w", encoding="utf-8") as f:

                # 5. 写入原始的用户问题 (来自 'question' 变量)
                f.write("========== 1. 原始用户问题 (User Question) ==========\n")
                f.write(question + "\n\n")

                # 6. 写入从图谱中检索到的初始上下文 (来自 'context_initial' 变量)
                f.write("========== 2. 检索到的初始图上下文 (Initial Graph Context) ==========\n")
                f.write(context_initial + "\n\n")

                # 7. 写入最终组合好的、即将发送给 LLM 的完整提示词 (来自 'init_prompt' 变量)
                f.write("========== 3. 完整的初始提示词 (Full Initial Prompt to LLM) ==========\n")
                f.write(init_prompt)

            # 8. (可选) 在服务器的控制台打印一条确认消息
            #    使用文件顶部已定义的 logger
            logger.info(f"[日志] 成功将初始提示词保存到: {filename}")

        except Exception as e:
            # 9. (可选) 如果保存失败，打印错误信息，但不会中断主程序
            logger.warning(f"[日志] 警告：保存提示词到 {filename} 失败: {e}")

        # ========== 新增代码保存提示词结束 (New Code Ends) ==========

        try:
            # Offload LLM call to thread executor
            initial_answer = await loop.run_in_executor(None, lambda: kt_retriever.generate_answer(init_prompt))
        except Exception as e:
            initial_answer = f"Initial answer failed: {e}"
        thoughts.append(f"Initial: {initial_answer[:200]}")
        final_answer = initial_answer

        # IRCoT 循环
        for step in range(1, max_steps + 1):
            loop_triples = _dedup(list(all_triples))
            loop_chunk_ids = list(set(all_chunk_ids))
            loop_chunk_contents = _merge_chunk_contents(loop_chunk_ids, all_chunk_contents)
            loop_ctx = "=== Triples ===\n" + "\n".join(loop_triples[:20]) + "\n=== Chunks ===\n" + "\n".join(
                loop_chunk_contents[:10])
            loop_prompt = f"""
You are an expert knowledge assistant using iterative retrieval with chain-of-thought reasoning.
Current Question: {question}
Current Iteration Query: {current_query}
Knowledge Context:\n{loop_ctx}
Previous Thoughts: {' | '.join(thoughts) if thoughts else 'None'}
Instructions:
1. If enough info answer with: So the answer is: <answer>
2. Else propose new query with: The new query is: <query>
Your reasoning:
"""
            try:
                reasoning = await loop.run_in_executor(None, lambda: kt_retriever.generate_answer(loop_prompt))
            except Exception as e:
                reasoning = f"Reasoning error: {e}"
            thoughts.append(reasoning[:400])
            reasoning_steps.append({
                "type": "ircot_step",
                "question": current_query,
                "triples": loop_triples[:10],
                "triples_count": len(loop_triples),
                "chunks_count": len(loop_chunk_ids),
                "processing_time": 0,
                "chunk_contents": loop_chunk_contents[:3],
                "thought": reasoning[:300]
            })

            # Stream iterative reasoning step updates (optional but helpful)
            try:
                await manager.send_message({
                    "type": "qa_update",
                    "stage": "ircot",
                    "step": step,
                    "max_steps": max_steps,
                    "current_query": current_query,
                    "thought_preview": (reasoning or "")[:200],
                    "timestamp": datetime.now().isoformat()
                }, client_id)
                # yield to event loop to flush WS frames
                await asyncio.sleep(0)
            except Exception as _e:
                logger.debug(f"QA update ws send failed for ircot step {step}: {_e}")
            if "So the answer is:" in reasoning:
                m = re.search(r"So the answer is:\s*(.*)", reasoning, flags=re.IGNORECASE | re.DOTALL)
                final_answer = m.group(1).strip() if m else reasoning
                break
            if "The new query is:" not in reasoning:
                final_answer = initial_answer or reasoning
                break
            new_query = reasoning.split("The new query is:", 1)[1].strip().splitlines()[0]
            if not new_query or new_query == current_query:
                final_answer = initial_answer or reasoning
                break
            current_query = new_query
            await send_progress_update(client_id, "retrieval", min(90, 75 + step * 5),
                                       f"Iterative retrieval step {step}...")
            try:
                def _run_more_retrieval():
                    return kt_retriever.process_retrieval_results(current_query, top_k=config.retrieval.top_k_filter)

                new_ret, _ = await loop.run_in_executor(None, _run_more_retrieval)
                new_triples = new_ret.get('triples', []) or []
                new_chunk_ids = new_ret.get('chunk_ids', []) or []
                new_chunk_contents = new_ret.get('chunk_contents', []) or []
                if isinstance(new_chunk_contents, dict):
                    for cid, ctext in new_chunk_contents.items():
                        all_chunk_contents[cid] = ctext
                else:
                    for i_c, cid in enumerate(new_chunk_ids):
                        if i_c < len(new_chunk_contents):
                            all_chunk_contents[cid] = new_chunk_contents[i_c]
                all_triples.update(new_triples)
                all_chunk_ids.update(new_chunk_ids)
            except Exception as e:
                logger.error(f"Iterative retrieval failed: {e}")
                break

        # Final aggregation
        final_triples = _dedup(list(all_triples))[:20]
        final_chunk_ids = list(set(all_chunk_ids))
        final_chunk_contents = _merge_chunk_contents(final_chunk_ids, all_chunk_contents)[:10]

        await send_progress_update(client_id, "retrieval", 100, "Answer generation completed!")

        # Notify frontend that QA process is complete with a compact summary
        try:
            await manager.send_message({
                "type": "qa_complete",
                "answer_preview": (final_answer or "")[:300],
                "sub_questions_count": len(sub_questions),
                "triples_final_count": len(final_triples),
                "chunks_final_count": len(final_chunk_contents),
                "timestamp": datetime.now().isoformat()
            }, client_id)
        except Exception as _e:
            logger.debug(f"QA complete ws send failed: {_e}")

        visualization_data = {
            "subqueries": prepare_subquery_visualization(sub_questions, reasoning_steps),
            "knowledge_graph": prepare_retrieved_graph_visualization(final_triples),
            "reasoning_flow": prepare_reasoning_flow_visualization(reasoning_steps),
            "retrieval_details": {
                "total_triples": len(final_triples),
                "total_chunks": len(final_chunk_contents),
                "sub_questions_count": len(sub_questions),
                "triples_by_subquery": [s.get("triples_count", 0) for s in reasoning_steps if
                                        s.get("type") == "sub_question"]
            }
        }

        return QuestionResponse(
            answer=final_answer,
            sub_questions=sub_questions,
            retrieved_triples=final_triples,
            retrieved_chunks=final_chunk_contents,
            reasoning_steps=reasoning_steps,
            visualization_data=visualization_data
        )
    except Exception as e:
        await send_progress_update(client_id, "retrieval", 0, f"Question answering failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def prepare_subquery_visualization(sub_questions: List[Dict], reasoning_steps: List[Dict]) -> Dict:
    """Prepare subquery visualization"""  # 准备子查询可视化：节点和链接
    nodes = [{"id": "original", "name": "Original Question", "category": "question", "symbolSize": 40}]
    links = []

    for i, sub_q in enumerate(sub_questions):
        sub_id = f"sub_{i}"
        nodes.append({
            "id": sub_id,
            "name": sub_q.get("sub-question", "")[:20] + "...",
            "category": "sub_question",
            "symbolSize": 30
        })
        links.append({"source": "original", "target": sub_id, "name": "decomposed to"})

    return {
        "nodes": nodes,
        "links": links,
        "categories": [
            {"name": "question", "itemStyle": {"color": "#ff6b6b"}},
            {"name": "sub_question", "itemStyle": {"color": "#4ecdc4"}}
        ]
    }


def prepare_retrieved_graph_visualization(triples: List[str]) -> Dict:
    """Prepare retrieved knowledge visualization"""  # 准备检索图可视化：基于三元组
    nodes = []
    links = []
    node_set = set()

    for triple in triples[:10]:
        try:
            if triple.startswith('[') and triple.endswith(']'):
                try:
                    parts = ast.literal_eval(triple)
                except Exception:
                    continue
                if len(parts) == 3:
                    source, relation, target = parts

                    for entity in [source, target]:
                        if entity not in node_set:
                            node_set.add(entity)
                            nodes.append({
                                "id": str(entity),
                                "name": str(entity)[:20],
                                "category": "entity",
                                "symbolSize": 20
                            })

                    links.append({
                        "source": str(source),
                        "target": str(target),
                        "name": str(relation)
                    })
        except:
            continue

    return {
        "nodes": nodes,
        "links": links,
        "categories": [{"name": "entity", "itemStyle": {"color": "#95de64"}}]
    }


def prepare_reasoning_flow_visualization(reasoning_steps: List[Dict]) -> Dict:
    """Prepare reasoning flow visualization"""  # 准备推理流可视化：步骤和时间线
    steps_data = []
    for i, step in enumerate(reasoning_steps):
        steps_data.append({
            "step": i + 1,
            "type": step.get("type", "unknown"),
            "question": step.get("question", "")[:50],
            "triples_count": step.get("triples_count", 0),
            "chunks_count": step.get("chunks_count", 0),
            "processing_time": step.get("processing_time", 0)
        })

    return {
        "steps": steps_data,
        "timeline": [step["processing_time"] for step in steps_data]
    }


@app.get("/api/datasets")
async def get_datasets(username: str = Depends(get_username)):
    """Get list of available datasets"""  # 获取数据集列表：列出用户上传和 demo 数据集，路径添加用户名隔离
    datasets = []

    # Check uploaded datasets
    upload_dir = f"data/uploaded/{username}"
    if os.path.exists(upload_dir):
        for item in os.listdir(upload_dir):
            item_path = os.path.join(upload_dir, item)
            if os.path.isdir(item_path):
                corpus_path = os.path.join(item_path, "corpus.json")
                if os.path.exists(corpus_path):
                    graph_path = f"output/graphs/{username}/{item}_new.json"
                    status = "ready" if os.path.exists(graph_path) else "needs_construction"
                    has_custom_schema = os.path.exists(f"schemas/{username}/{item}.json")
                    datasets.append({
                        "name": item,
                        "type": "uploaded",
                        "status": status,
                        "has_custom_schema": has_custom_schema
                    })

    # Add demo dataset
    demo_corpus = "data/demo/demo_corpus.json"
    if os.path.exists(demo_corpus):
        demo_graph = "output/graphs/demo_new.json"
        status = "ready" if os.path.exists(demo_graph) else "needs_construction"
        datasets.append({
            "name": "demo",
            "type": "demo",
            "status": status,
            "has_custom_schema": False
        })

    return {"datasets": datasets}


@app.post("/api/datasets/{dataset_name}/schema")
async def upload_schema(
        dataset_name: str,
        schema_file: UploadFile = File(...),
        username: str = Depends(get_username)
):
    """Upload a custom schema JSON for a dataset."""  # 上传 schema 端点：保存用户特定 schema，路径添加用户名隔离
    try:
        if dataset_name == "demo":
            raise HTTPException(status_code=400, detail="Cannot upload schema for demo dataset")
        if not schema_file.filename.lower().endswith('.json'):
            raise HTTPException(status_code=400, detail="Schema file must be a .json file")

        content = await schema_file.read()
        try:
            schema_text = decode_bytes_with_detection(content)
            data = json.loads(schema_text)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="Schema JSON must be an object")

        os.makedirs(f"schemas/{username}", exist_ok=True)
        save_path = f"schemas/{username}/{dataset_name}.json"
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return {"success": True, "message": "Schema uploaded successfully", "dataset_name": dataset_name}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload schema: {str(e)}")


@app.delete("/api/datasets/{dataset_name}")
async def delete_dataset(
        dataset_name: str,
        username: str = Depends(get_username)
):
    """Delete a dataset and all its associated files"""  # 删除数据集端点：移除用户特定文件和缓存，路径添加用户名隔离
    try:
        if dataset_name == "demo":
            raise HTTPException(status_code=400, detail="Cannot delete demo dataset")

        deleted_files = []

        # Delete dataset directory
        dataset_dir = f"data/uploaded/{username}/{dataset_name}"
        if os.path.exists(dataset_dir):
            import shutil
            shutil.rmtree(dataset_dir)
            deleted_files.append(dataset_dir)

        # Delete graph file
        graph_path = f"output/graphs/{username}/{dataset_name}_new.json"
        if os.path.exists(graph_path):
            os.remove(graph_path)
            deleted_files.append(graph_path)

        # Delete schema file (if dataset-specific)
        schema_path = f"schemas/{username}/{dataset_name}.json"
        if os.path.exists(schema_path):
            os.remove(schema_path)
            deleted_files.append(schema_path)

        # Delete cache files
        cache_dir = f"retriever/faiss_cache_new/{username}/{dataset_name}"
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
            deleted_files.append(cache_dir)

        # Delete chunk files
        chunk_file = f"output/chunks/{username}/{dataset_name}.txt"
        if os.path.exists(chunk_file):
            os.remove(chunk_file)
            deleted_files.append(chunk_file)

        return {
            "success": True,
            "message": f"Dataset '{dataset_name}' deleted successfully",
            "deleted_files": deleted_files
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {str(e)}")


@app.post("/api/datasets/{dataset_name}/reconstruct")
async def reconstruct_dataset(
        dataset_name: str,
        client_id: str = "default",
        username: str = Depends(get_username)
):
    """Reconstruct graph for an existing dataset"""  # 重建图端点：清理旧图和缓存，重新构建，路径添加用户名隔离
    try:
        if not GRAPHRAG_AVAILABLE:
            raise HTTPException(status_code=503,
                                detail="GraphRAG components not available. Please install or configure them.")
        # Check if dataset exists
        corpus_path = f"data/uploaded/{username}/{dataset_name}/corpus.json"
        if not os.path.exists(corpus_path):
            if dataset_name == "demo":
                corpus_path = "data/demo/demo_corpus.json"
            else:
                raise HTTPException(status_code=404, detail="Dataset not found")

        await send_progress_update(client_id, "reconstruction", 5, "Starting reconstruction...")

        # Delete existing graph file
        graph_path = f"output/graphs/{username}/{dataset_name}_new.json"
        if os.path.exists(graph_path):
            os.remove(graph_path)
            await send_progress_update(client_id, "reconstruction", 15, "Old graph file deleted...")

        # Delete existing cache files
        cache_dir = f"retriever/faiss_cache_new/{username}/{dataset_name}"
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
            await send_progress_update(client_id, "reconstruction", 25, "Cache files cleared...")

        await send_progress_update(client_id, "reconstruction", 35, "Reinitializing graph builder...")

        # Initialize config
        global config
        if config is None:
            config = get_config("config/base_config.yaml")

        # Choose schema: dataset-specific or default demo
        schema_path = get_schema_path_for_dataset(username, dataset_name)

        # Initialize KTBuilder
        builder = constructor.KTBuilder(
            dataset_name,
            schema_path,
            mode=config.construction.mode,
            config=config
        )

        await send_progress_update(client_id, "reconstruction", 50, "Rebuilding knowledge graph...")

        # Build knowledge graph
        def build_graph_sync():
            return builder.build_knowledge_graph(corpus_path)

        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()

        # Run graph reconstruction without simulated progress updates
        knowledge_graph = await loop.run_in_executor(None, build_graph_sync)

        await send_progress_update(client_id, "reconstruction", 100, "Graph reconstruction completed!")
        # Notify completion via WebSocket
        try:
            await manager.send_message({
                "type": "complete",
                "stage": "reconstruction",
                "message": "Graph reconstruction completed!",
                "timestamp": datetime.now().isoformat()
            }, client_id)
        except Exception as _e:
            logger.warning(f"Failed to send completion message: {_e}")

        return {
            "success": True,
            "message": "Dataset reconstructed successfully",
            "dataset_name": dataset_name
        }

    except Exception as e:
        await send_progress_update(client_id, "reconstruction", 0, f"Reconstruction failed: {str(e)}")
        try:
            await manager.send_message({
                "type": "error",
                "stage": "reconstruction",
                "message": f"Reconstruction failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }, client_id)
        except Exception as _e:
            logger.warning(f"Failed to send error message: {_e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/graph/{dataset_name}")
async def get_graph_data(
        dataset_name: str,
        username: str = Depends(get_username)
):
    """Get graph visualization data"""  # 获取图可视化数据：加载用户特定图文件
    graph_path = f"output/graphs/{username}/{dataset_name}_new.json"

    if not os.path.exists(graph_path):
        # Return demo data
        return {
            "nodes": [
                {"id": "node1", "name": "Example Entity 1", "category": "person", "value": 5, "symbolSize": 25},
                {"id": "node2", "name": "Example Entity 2", "category": "location", "value": 3, "symbolSize": 20},
            ],
            "links": [
                {"source": "node1", "target": "node2", "name": "located_in", "value": 1}
            ],
            "categories": [
                {"name": "person", "itemStyle": {"color": "#ff6b6b"}},
                {"name": "location", "itemStyle": {"color": "#4ecdc4"}},
            ],
            "stats": {"total_nodes": 2, "total_edges": 1, "displayed_nodes": 2, "displayed_edges": 1}
        }

    return await prepare_graph_visualization(graph_path)


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""  # 启动事件：创建目录 + Nacos 注册
    os.makedirs("data/uploaded", exist_ok=True)
    os.makedirs("output/graphs", exist_ok=True)
    os.makedirs("output/logs", exist_ok=True)
    os.makedirs("schemas", exist_ok=True)

    # Nacos 注册（修正版）
    service_name = 'youtu-graphrag-api'
    hostname = socket.gethostname()
    service_ip = socket.gethostbyname(hostname)
    service_port = int(os.getenv('PORT', 8000))
    group_name = os.getenv('NACOS_GROUP', 'DEFAULT_GROUP')  # 移到这里
    service_ip = '127.0.0.1'  # 本地固定
    service_port = 8000  # 固定端口
    group_name = 'DEFAULT_GROUP'  # 固定组
    try:
        client.add_naming_instance(  # 1.0.0 方法
            service_name,
            service_ip,
            service_port,
            group_name=group_name,  # 在方法中传入
            weight=1.0
        )
        logger.info(f"Service '{service_name}' registered to Nacos at {service_ip}:{service_port}")
    except NacosException as e:
        logger.error(f"Nacos registration failed: {e}")

    logger.info("🚀 Youtu-GraphRAG Unified Interface initialized")


@app.on_event("shutdown")
def shutdown_event():  # 同步
    service_name = os.getenv('SERVICE_NAME', 'youtu-graphrag-api')
    hostname = socket.gethostname()
    service_ip = socket.gethostbyname(hostname)
    service_port = int(os.getenv('PORT', 8000))
    group_name = os.getenv('NACOS_GROUP', 'DEFAULT_GROUP')

    try:
        client.remove_naming_instance(  # 1.0.0 方法
            service_name,
            service_ip,
            service_port,
            group_name=group_name
        )
        logger.info(f"Service '{service_name}' deregistered from Nacos")
    except NacosException as e:
        logger.error(f"Nacos deregistration failed: {e}")


if __name__ == "__main__":
    # 配置 uvicorn 日志格式，添加时间戳
    from uvicorn.config import LOGGING_CONFIG

    LOGGING_CONFIG["formatters"]["default"]["fmt"] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOGGING_CONFIG["formatters"]["access"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
    LOGGING_CONFIG["formatters"]["default"]["datefmt"] = "%Y-%m-%d %H:%M:%S"
    LOGGING_CONFIG["formatters"]["access"]["datefmt"] = "%Y-%m-%d %H:%M:%S"
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=LOGGING_CONFIG)

    # 调用接口顺序：
    #