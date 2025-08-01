# main.py - GraphRAG主程序
import os
import json
import re
import subprocess
from typing import List, Dict, Any, Optional, Union
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import requests
from neo4j import GraphDatabase
import tiktoken
import uuid
from datetime import datetime
import logging
from PyPDF2 import PdfReader
import time

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置类 - 适配Windows环境
class Config:
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "password"
    OLLAMA_BASE_URL = "http://localhost:11434"
    EMBEDDING_MODEL = "nomic-embed-text"
    LLM_MODEL = "gemma3:12b"
    MAX_CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    UPLOAD_DIR = "uploads"

# 确保上传目录存在
os.makedirs(Config.UPLOAD_DIR, exist_ok=True)

# 数据模型
class DocumentInput(BaseModel):
    title: str
    content: str
    metadata: Dict[str, Any] = {}

class QueryInput(BaseModel):
    query: str
    limit: int = 10
    include_graph: bool = True

class Entity(BaseModel):
    name: str
    type: Optional[str] = "Entity"
    metadata: Dict[str, Any] = {}

class Relationship(BaseModel):
    source: str
    target: str
    type: Optional[str] = "Relationship"
    properties: Dict[str, Any] = {}

# Ollama客户端 - 增强错误处理
class OllamaClient:
    def __init__(self, base_url: str = Config.OLLAMA_BASE_URL):
        self.base_url = base_url
    
    def check_connection(self) -> bool:
        """检查Ollama连接"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate(self, model: str, prompt: str, **kwargs) -> str:
        """生成文本"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        **kwargs
                    }
                },
                timeout=600
            )
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama generate error: {e}")
            raise HTTPException(status_code=500, detail=f"Ollama服务异常: {str(e)}")
    
    def embed(self, model: str, input_text: str) -> List[float]:
        """生成文本嵌入"""
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": model,
                    "prompt": input_text
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama embed error: {e}")
            raise HTTPException(status_code=500, detail=f"嵌入生成失败: {str(e)}")

# Neo4j数据库管理器 - 增强连接处理
class Neo4jManager:
    def __init__(self, uri: str, user: str, password: str, parent=None):
        parent = parent 
        max_retries = 2  # 最大重试次数
        retry_delay = 5  # 重试间隔(秒)
        
        for attempt in range(max_retries + 1):
            try:
                self.driver = GraphDatabase.driver(uri, auth=(user, password))
                # 测试连接
                with self.driver.session() as session:
                    session.run("RETURN 1")
                logger.info("Neo4j连接成功")
                return  # 连接成功，直接返回
                
            except Exception as e:
                if attempt == max_retries:
                    logger.error(f"Neo4j连接失败，已达到最大重试次数: {e}")
                    raise HTTPException(status_code=500, detail=f"数据库连接失败: {str(e)}")
                
                logger.warning(f"Neo4j连接失败，尝试启动容器 (尝试 {attempt + 1}/{max_retries}): {e}")
                self._start_neo4j_container(user, password)
                time.sleep(retry_delay)  # 等待容器启动
    
    def _start_neo4j_container(self, user: str, password: str):
        """尝试启动Neo4j容器（使用系统docker命令）"""
        container_name = "neo4j"
        
        try:
            # 检查容器是否存在
            check_container_cmd = ["docker", "ps", "-a", "--filter", f"name={container_name}", "--format", "{{.Status}}"]
            result = subprocess.run(check_container_cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout.strip():
                status = result.stdout.strip().split()[0]
                if status != "Up":
                    logger.info(f"启动已存在的Neo4j容器: {container_name}")
                    subprocess.run(["docker", "start", container_name], check=True)
                else:
                    logger.info(f"Neo4j容器已在运行: {container_name}")
                return
            
            # 检查镜像是否存在
            check_image_cmd = ["docker", "images", "-q", "neo4j:latest"]
            result = subprocess.run(check_image_cmd, capture_output=True, text=True)
            
            if not result.stdout.strip():
                logger.info("拉取Neo4j最新镜像...")
                subprocess.run(["docker", "pull", "neo4j:latest"], check=True)
            
            # 运行新容器
            logger.info("创建并启动新的Neo4j容器")
            subprocess.run([
                "docker", "run", "-d",
                "--name", container_name,
                "-p", "7474:7474",
                "-p", "7687:7687",
                "-e", f"NEO4J_AUTH={user}/{password}",
                "neo4j:latest"
            ], check=True)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Docker命令执行失败: {e}")
            raise HTTPException(status_code=500, detail=f"Docker操作失败: {str(e)}")
        except FileNotFoundError:
            logger.error("Docker未安装或不在PATH中")
            raise HTTPException(status_code=500, detail="Docker未安装或不可用")
           
    def close(self):
        if hasattr(self, 'driver'):
            self.driver.close()
    
    def create_indexes(self):
        """创建必要的索引"""
        try:
            with self.driver.session() as session:
                session.run("CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.id)")
                session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)")
                session.run("CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.id)")
                logger.info("索引创建完成")
        except Exception as e:
            logger.error(f"索引创建失败: {e}")
    
    def store_document(self, doc_id: str, title: str, content: str, metadata: Dict):
        """存储文档"""
        with self.driver.session() as session:
            session.run(
                """
                MERGE (d:Document {id: $doc_id})
                SET d.title = $title, d.content = $content, d.metadata = $metadata, d.created_at = $timestamp
                """,
                doc_id=doc_id, title=title, content=content, 
                metadata=json.dumps(metadata), timestamp=datetime.now().isoformat()
            )
    
    def store_chunk(self, chunk_id: str, doc_id: str, content: str, embedding: List[float], metadata: Dict[str, Any]):
        """存储文档块及其元信息"""
        with self.driver.session() as session:
            session.run(
                """
                MERGE (c:Chunk {id: $chunk_id})
                SET c.content = $content, 
                    c.embedding = $embedding,
                    c.metadata = $metadata
                WITH c
                MATCH (d:Document {id: $doc_id})
                MERGE (d)-[:HAS_CHUNK]->(c)
                """,
                chunk_id=chunk_id,
                doc_id=doc_id,
                content=content,
                embedding=embedding,
                metadata=json.dumps(metadata)
            )
            
    def store_chunks(self, chunks: List[Dict], doc_id: str):
        """批量存储所有分块"""
        if not graph_rag:
            raise ValueError("GraphRAG 未初始化")
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{idx}"
            embedding = graph_rag.ollama_client.embed(Config.EMBEDDING_MODEL, chunk["text"])  # 假设你已有 embedder
            self.store_chunk(chunk_id, doc_id, chunk["text"], embedding, chunk.get("metadata", {}))

    def store_entitie(self, entity: Entity, doc_id: str):
        """存储实体及其元信息"""
        with self.driver.session() as session:
            session.run(
                    """
                    MERGE (e:Entity {name: $name})
                    SET e.type = coalesce(e.type, $type),
                        e += $properties
                    WITH e
                    MATCH (d:Document {id: $doc_id})
                    MERGE (d)-[:CONTAINS_ENTITY]->(e)
                    """,
                    name=entity.name,
                    type=entity.type,
                    properties=entity.metadata or {},
                    doc_id=doc_id
                )

        
    def store_entities(self, entities: List[Entity], doc_id: str):
        """存储实体并与文档关联"""
        with self.driver.session() as session:
            for entity in entities:
                self.store_entitie(entity, doc_id)

    def store_relationships(self, relationships: List[Relationship], doc_id: str):
        """存储关系"""
        with self.driver.session() as session:
            for rel in relationships:
                session.run(
                    """
                    MATCH (s:Entity {name: $source}), (t:Entity {name: $target})
                    MERGE (s)-[r:RELATED {type: $type}]->(t)
                    SET r.properties = $properties,
                        r.doc_id = $doc_id
                    """,
                    source=rel.source, target=rel.target, type=rel.type,
                    properties=json.dumps(rel.properties), doc_id=doc_id
                )
    
    def search_similar_chunks(self, query_embedding: List[float], limit: int = 10):
        """搜索相似文档块 - 使用欧几里德距离"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (c:Chunk)
                WHERE c.embedding IS NOT NULL
                WITH c, 
                    reduce(dot = 0.0, i IN range(0, size($query_embedding)-1) | 
                            dot + (c.embedding[i] * $query_embedding[i])) as dot_product,
                    sqrt(reduce(norm_a = 0.0, a IN c.embedding | norm_a + a^2)) as norm_a,
                    sqrt(reduce(norm_b = 0.0, b IN $query_embedding | norm_b + b^2)) as norm_b
                WHERE norm_a > 0 AND norm_b > 0
                WITH c, dot_product / (norm_a * norm_b) as similarity
                MATCH (d:Document)-[:HAS_CHUNK]->(c)
                RETURN c.id as chunk_id, c.content as content, similarity, d.title as title, c.metadata as metadata
                ORDER BY similarity DESC
                LIMIT $limit
                """,
                query_embedding=query_embedding, limit=limit
            )
            return [dict(record) for record in result]
    
    def get_entity_subgraph(self, entity_names: List[str]):
        """获取实体子图"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (e1:Entity)-[r]-(e2:Entity)
                WHERE e1.name IN $entity_names OR e2.name IN $entity_names
                RETURN e1.name as source, e2.name as target, type(r) as relationship
                LIMIT 50
                """,
                entity_names=entity_names
            )
            return [dict(record) for record in result]
    
    def get_stats(self):
        """获取统计信息"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (d:Document) 
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                OPTIONAL MATCH (d)-[:CONTAINS_ENTITY]->(e:Entity)
                OPTIONAL MATCH ()-[r:RELATED]->()
                RETURN 
                    count(DISTINCT d) as documents,
                    count(DISTINCT c) as chunks,
                    count(DISTINCT e) as entities,
                    count(DISTINCT r) as relationships
                """
            )
            r = result.single()
            return dict(r if r else [] )

# 文本处理器 - 优化实体提取
class TextProcessor:
    def __init__(self, ollama_client: OllamaClient):
        self.ollama_client = ollama_client
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            # 如果tiktoken不可用，使用简单的分词
            self.tokenizer = None
    
    def chunk_text(self, text: str, max_size: int = Config.MAX_CHUNK_SIZE, 
                   overlap: int = Config.CHUNK_OVERLAP) -> List[str]:
        """将文本分块"""
        if self.tokenizer:
            tokens = self.tokenizer.encode(text)
            chunks = []
            
            start = 0
            while start < len(tokens):
                end = min(start + max_size, len(tokens))
                chunk_tokens = tokens[start:end]
                chunk_text = self.tokenizer.decode(chunk_tokens)
                chunks.append(chunk_text)
                
                if end == len(tokens):
                    break
                start = end - overlap
            
            return chunks
        else:
            # 简单的字符分块
            words = text.split()
            chunks = []
            chunk_size = max_size // 4  # 估算单词数
            
            for i in range(0, len(words), chunk_size - overlap // 4):
                chunk_words = words[i:i + chunk_size]
                chunks.append(' '.join(chunk_words))
            
            return chunks
    
    def extract_entities_and_relations(self, text: str) -> tuple[List[Entity], List[Relationship]]:
        """从文本中提取实体和关系"""
        if len(text.strip()) < 20:  # 跳过太短的文本
            return [], []
            
        prompt = f"""
请从以下文本中提取重要的实体和它们之间的关系。

要求：
1. 只提取重要的人名、地名、组织、概念等实体
2. 只提取明确的关系，不要推断
3. 返回有效的JSON格式
4. 使用原文档的语言和格式，不要翻译或修改实体名称

示例格式：
{{
    "entities": [
        {{"name": "张三", "type": "人物", "properties": {{}}}},
        {{"name": "北京", "type": "地点", "properties": {{}}}}
    ],
    "relationships": [
        {{"source": "张三", "target": "北京", "type": "居住在", "properties": {{}}}}
    ]
}}

文本：
{text[:500]}...

JSON结果：
"""
        
        try:
            response = self.ollama_client.generate(Config.LLM_MODEL, prompt)
            # 尝试提取JSON部分
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                
                entities = []
                for entity_data in result.get("entities", [])[:10]:  # 限制数量
                    if isinstance(entity_data, dict) and entity_data.get("name"):
                        entities.append(Entity(**entity_data))
                
                relationships = []
                for rel_data in result.get("relationships", [])[:10]:  # 限制数量
                    if (isinstance(rel_data, dict) and 
                        rel_data.get("source") and rel_data.get("target")):
                        relationships.append(Relationship(**rel_data))
                
                return entities, relationships
            
        except Exception as e:
            logger.warning(f"实体提取失败: {e}")
        
        return [], []

# GraphRAG主类
class GraphRAG:
    def __init__(self):
        self.ollama_client = OllamaClient()
        self.neo4j_manager = None
        self.text_processor = TextProcessor(self.ollama_client)
        
        # 检查服务可用性
        if not self.ollama_client.check_connection():
            logger.warning("Ollama服务未启动，请确保运行 'ollama serve'")
        
        try:
            self.neo4j_manager = Neo4jManager(
                Config.NEO4J_URI, Config.NEO4J_USER, Config.NEO4J_PASSWORD
            )
            self.neo4j_manager.create_indexes()
        except Exception as e:
            logger.error(f"Neo4j初始化失败: {e}")
    
    def add_document(self, title: str, paragraphs: List[Dict], metadata: Dict[str, Any] = {}):
        """添加文档到知识库"""
        if not self.neo4j_manager:
            raise HTTPException(status_code=500, detail="数据库未连接")
            
        if metadata is None:
            metadata = {}
        
        doc_id = str(uuid.uuid4())
        
        try:
            content = "\n\n".join(p["text"] for p in paragraphs)
            # 存储原始文档
            self.neo4j_manager.store_document(doc_id, title, content, metadata)
            
            # 分块处理
            chunk_process(paragraphs, doc_id)
            return doc_id
        except Exception as e:
            logger.error(f"文档添加失败: {e}")
            raise HTTPException(status_code=500, detail=f"文档添加失败: {str(e)}")

    def safe_load_metadata(self, chunk):
        try:
            return json.loads(chunk.get('metadata', '{}'))
        except json.JSONDecodeError:
            return {}
            
    def query(self, query: str, limit: int = 10, include_graph: bool = True):
        """查询知识库"""
        if not self.neo4j_manager:
            raise HTTPException(status_code=500, detail="数据库未连接")
        
        try:
            # 生成查询嵌入
            query_embedding = self.ollama_client.embed(Config.EMBEDDING_MODEL, query)
            
            # 搜索相似文档块
            similar_chunks = self.neo4j_manager.search_similar_chunks(query_embedding, limit)
            
            if not similar_chunks:
                return {
                    "answer": "抱歉，在知识库中没有找到相关信息。",
                    "sources": [],
                    "query": query,
                    "graph": []
                }
            
            # 构建上下文
            context = "\n\n".join([
                f"文档片段 {i+1}（文件: {metadata.get('filename', '未知')}，"
                f"页码: {metadata.get('page', '?')}，"
                f"段落号: {metadata.get('paragraph_index', '?')}）:\n"
                f"{chunk['content']}"
                for i, chunk in enumerate(similar_chunks[:5])
                for metadata in [self.safe_load_metadata(chunk)]  # 安全解析
            ])
            
            # 生成回答
            prompt = f"""
基于以下上下文信息回答用户问题，要求：
1. 回答要准确、具体、有用，内容必须基于文档片段，不要凭空想象。
2. 如果上下文中没有相关信息，请明确说明
3. 用文档的语言回答

上下文信息：
{context}

用户问题：{query}

回答：
"""
            
            answer = self.ollama_client.generate(Config.LLM_MODEL, prompt)
            
            result = {
                "answer": answer,
                "sources": similar_chunks,
                "query": query
            }
            
            # 如果需要图信息
            if include_graph and similar_chunks:
                # 提取问题中的实体
                entity_prompt = f"从这个问题中提取主要的实体名词（人名、地名、组织名等），只返回实体名称，用逗号分隔，并保持原有的语言不要翻译或修改实体名称：{query}"
                entities_response = self.ollama_client.generate(Config.LLM_MODEL, entity_prompt)
                entity_names = [name.strip() for name in entities_response.split(",")[:5]]
                
                subgraph = self.neo4j_manager.get_entity_subgraph(entity_names)
                result["graph"] = subgraph
            else:
                result["graph"] = []
            
            return result
            
        except Exception as e:
            logger.error(f"查询失败: {e}")
            raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")

# FastAPI应用
app = FastAPI(title="GraphRAG Knowledge Base", version="1.0.0", description="基于Ollama的GraphRAG知识库系统")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件服务
app.mount("/static", StaticFiles(directory="static"), name="static")

# 添加管理页面路由
@app.get("/manage")
async def manage_page():
    return FileResponse("static/manage.html")

# 添加管理页面路由
@app.get("/manage_old")
async def manage_old_page():
    return FileResponse("static/manage_old.html")

# 初始化GraphRAG
graph_rag = None

@app.on_event("startup")
async def startup_event():
    global graph_rag
    try:
        graph_rag = GraphRAG()
        logger.info("GraphRAG系统启动完成")
    except Exception as e:
        logger.error(f"系统启动失败: {e}")

@app.get("/")
async def read_root():
    """主页"""
    return FileResponse("static/index.html")

@app.get("/style.css")
async def read_css():
    """css"""
    return FileResponse("static/style.css")

@app.post("/api/documents/add")
async def add_document(document: DocumentInput):
    """添加文档"""
    if not graph_rag:
        raise HTTPException(status_code=500, detail="系统未初始化")
        
    try:
        doc_id = graph_rag.add_document(
            title=document.title,
            paragraphs=process_text_file(document.content),
            metadata=document.metadata
        )
        return {"doc_id": doc_id, "message": "文档添加成功"}
    except Exception as e:
        logger.error(f"文档添加失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download")
async def download_file(filename: str):
    file_path = os.path.join(Config.UPLOAD_DIR, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail=f"文件: {file_path} 不存在")
    return FileResponse(file_path, filename=filename, media_type='application/pdf')

@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """上传文档文件"""
    if not graph_rag:
        raise HTTPException(status_code=500, detail="系统未初始化")
    
    try:
        content = await file.read()
        filename = file.filename or "No filename"
        content_type = file.content_type or "no type"
        filetype = filename.lower()
        if filetype.endswith('.pdf'):
            paragraphs = process_pdf_file(content)
        elif filetype.endswith('.txt'):
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                raise HTTPException(status_code=400, detail="文本文件编码不支持，请使用 UTF-8")
            paragraphs = process_text_file(text)
        else:
            raise HTTPException(status_code=400, detail="仅支持 .pdf 和 .txt 文件")
        
        file_path = os.path.join(Config.UPLOAD_DIR, filename)
        with open(file_path, "wb") as f:
            f.write(content)
        logger.info(f"文件已保存到 {file_path}")        
        doc_id = graph_rag.add_document(
            title=filename or "No filename",
            paragraphs=paragraphs,
            metadata={"filename": filename, "content_type": content_type}
        ) if graph_rag else ""
        
        return {"doc_id": doc_id, "message": "文件上传成功"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        raise HTTPException(status_code=500, detail="文件处理时发生错误")
    
#TODO:use other models to help denoise
def process_text_file(text: str, page: int = 0) -> List[Dict]:
    """按段落切分纯文本，附加页面信息（默认为 0）"""
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    raw_paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    return [
        {
            "text": para,
            "metadata": {
                "page": page,
                "paragraph_index": idx
            }
        }
        for idx, para in enumerate(raw_paragraphs)
    ]

def process_pdf_file(content: bytes) -> List[Dict]:
    """按段落切分 PDF，每页提取文本，并附加 page 信息"""
    try:
        import io
        pdf_stream = io.BytesIO(content)
        reader = PdfReader(pdf_stream)
        paragraphs = []
        for page_number, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            paras = process_text_file(text, page=page_number)
            paragraphs.extend(paras)
        return paragraphs
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无法读取 PDF 内容: {e}")

#TODO:use other models to help denoise


@app.post("/api/query")
async def query_knowledge_base(query_input: QueryInput):
    """查询知识库"""
    if not graph_rag:
        raise HTTPException(status_code=500, detail="系统未初始化")
        
    try:
        result = graph_rag.query(
            query=query_input.query,
            limit=query_input.limit,
            include_graph=query_input.include_graph
        )
        return result
    except Exception as e:
        logger.error(f"查询失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """健康检查"""
    status = {
        "status": "healthy" if graph_rag else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "services": {}
    }
    
    if graph_rag:
        # 检查Ollama连接
        status["services"]["ollama"] = graph_rag.ollama_client.check_connection()
        
        # 检查Neo4j连接
        try:
            if graph_rag.neo4j_manager:
                with graph_rag.neo4j_manager.driver.session() as session:
                    session.run("RETURN 1")
                status["services"]["neo4j"] = True
            else:
                status["services"]["neo4j"] = False
        except:
            status["services"]["neo4j"] = False
    
    return status

@app.get("/api/stats")
async def get_stats():
    """获取统计信息"""
    if not graph_rag or not graph_rag.neo4j_manager:
        raise HTTPException(status_code=500, detail="系统未初始化")
    
    try:
        stats = graph_rag.neo4j_manager.get_stats()
        return stats
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class DeleteDocumentInput(BaseModel):
    doc_id: str

class ResetDatabaseInput(BaseModel):
    confirm: bool

class RebuildIndexInput(BaseModel):
    doc_id: str



@app.get("/api/documents/list")
async def list_documents(page: int = 1, per_page: int = 8, search: str = ""):
    """获取文档列表（分页+搜索）"""
    if not graph_rag or not graph_rag.neo4j_manager:
        raise HTTPException(status_code=500, detail="系统未初始化")
    
    try:
        skip = (page - 1) * per_page
        query = """
            MATCH (d:Document)
            WHERE $search = "" OR d.title CONTAINS $search OR d.content CONTAINS $search
            WITH d
            ORDER BY d.created_at DESC
            SKIP $skip LIMIT $limit

            CALL {
                WITH d
                RETURN COUNT { (d)-[:CONTAINS_ENTITY]->(:Entity) } AS entity_count
            }
            CALL {
                WITH d
                MATCH ()-[r:RELATED]->()
                WHERE r.doc_id = d.id
                RETURN count(r) AS relationship_count
            }

            RETURN d.id AS id,
                d.title AS title,
                substring(d.content, 0, 100) AS preview,
                d.created_at AS created_at,
                entity_count,
                relationship_count
        """
        with graph_rag.neo4j_manager.driver.session() as session:
            result = session.run(query, search=search, skip=skip, limit=per_page)
            documents = [dict(record) for record in result]
            
            # 获取总数
            count_result = session.run(
                "MATCH (d:Document) RETURN count(d) as total"
            )
            t = count_result.single()
            total = t["total"] if t else 0
            
        return {
            "documents": documents,
            "total": total,
            "page": page,
            "per_page": per_page
        }
    except Exception as e:
        logger.error(f"文档列表获取失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
#使用/api/relationships/rebuild来重建单个文档内的关系
@app.post("/api/relationships/rebuild")
async def rebuild_document_relationships(data: RebuildIndexInput):
    """重建单个文档内的实体和关系"""
    if not graph_rag or not graph_rag.neo4j_manager:
        raise HTTPException(status_code=500, detail="系统未初始化")

    try:
        with graph_rag.neo4j_manager.driver.session() as session:
            # 1. 获取文档内容
            result = session.run(
                """
                MATCH (d:Document {id: $doc_id})
                RETURN d.content AS content
                """,
                doc_id=data.doc_id
            )
            record = result.single()
            if not record:
                raise HTTPException(status_code=404, detail="找不到指定文档")
            content = record["content"]

            # 2. 删除旧的 CONTAINS_ENTITY 和 RELATED 关系 + 独占实体
            session.run(
                """
                // 删除与该文档的实体连接
                MATCH (d:Document {id: $doc_id})-[r:CONTAINS_ENTITY]->(e:Entity)
                DELETE r
                WITH $doc_id AS doc_id

                // 删除与该文档的关系连接
                MATCH ()-[rel:RELATED {doc_id: doc_id}]->()
                DELETE rel
                WITH doc_id

                // 删除该文档唯一引用的实体
                MATCH (e:Entity)
                WHERE NOT (e)<-[:CONTAINS_ENTITY]-(:Document)
                DETACH DELETE e
                """,
                doc_id=data.doc_id
            )

        # 3. 分块并重新提取实体关系并返回
        return {"success": chunk_process(content,data.doc_id,True), "message": "文档实体和关系已重建"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文档实体和关系重建失败: {e}") 
        raise HTTPException(status_code=500, detail=str(e))
    
def chunk_process(paragraphs: List[Dict], doc_id: str, rebuider=False):
    if not graph_rag or not graph_rag.neo4j_manager:
        raise HTTPException(status_code=500, detail="系统未初始化")

    chunks = []
    for i in range(len(paragraphs)):
        window = [paragraphs[j] for j in range(i - 1, i + 2) if 0 <= j < len(paragraphs)]
        chunk_text = "\n\n".join(p["text"] for p in window)
        chunk_meta = window[1].get("metadata", {}) if len(window) > 1 else {}
        chunk_id = f"{doc_id}_{i}"
        embedding = graph_rag.ollama_client.embed(Config.EMBEDDING_MODEL,chunk_text)
        chunks.append({
            "text": chunk_text,
            "metadata": chunk_meta
        })
        graph_rag.neo4j_manager.store_chunk(chunk_id, doc_id, chunk_text, embedding, chunk_meta)

        if i % 2 != 0:
            continue  # 隔一个处理
        entities, relationships = graph_rag.text_processor.extract_entities_and_relations(chunk_text)
        if entities:
            for ent in entities:
                ent.metadata = {**ent.metadata, **chunk_meta} if hasattr(ent, "metadata") else chunk_meta
            graph_rag.neo4j_manager.store_entities(entities, doc_id)
            logger.info(f"{'[重建索引]' if rebuider else ''} 第{i}块提取到 {len(entities)} 个实体")

        if relationships:
            graph_rag.neo4j_manager.store_relationships(relationships, doc_id)
            logger.info(f"{'[重建索引]' if rebuider else ''} 第{i}块提取到 {len(relationships)} 个关系")
    logger.info(f"{'[重建索引]' if rebuider else ''} 文档被分为 {len(chunks)} 块")
    return True


@app.post("/api/documents/delete")
async def delete_document(data: DeleteDocumentInput):
    """删除文档及其所有关联数据"""
    if not graph_rag or not graph_rag.neo4j_manager:
        raise HTTPException(status_code=500, detail="系统未初始化")
    
    try:
        with graph_rag.neo4j_manager.driver.session() as session:
            # 删除文档及其所有关联节点
            session.run(
                """
                MATCH (d:Document {id: $doc_id})
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                OPTIONAL MATCH (d)-[:CONTAINS_ENTITY]->(e:Entity)
                DETACH DELETE d, c, e
                """,
                doc_id=data.doc_id
            )
        return {"success": True, "message": "文档已删除"}
    except Exception as e:
        logger.error(f"文档删除失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/database/reset")
async def reset_database(data: ResetDatabaseInput):
    """清空整个数据库（危险操作）"""
    if not data.confirm:
        raise HTTPException(status_code=400, detail="需要确认参数")
    
    try:
        if graph_rag and graph_rag.neo4j_manager:
            with graph_rag.neo4j_manager.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            return {"success": True, "message": "数据库已重置"}
    except Exception as e:
        logger.error(f"数据库重置失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/graph/visualization")
async def get_visualization_data(limit: int = 50):
    """获取可视化图数据"""
    try:
        if graph_rag and graph_rag.neo4j_manager:
            with graph_rag.neo4j_manager.driver.session() as session:
                # 获取文档-实体-关系的子图
                result = session.run(
                    """
                    MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
                    OPTIONAL MATCH (d)-[:CONTAINS_ENTITY]->(e:Entity)
                    OPTIONAL MATCH (e1:Entity)-[r:RELATED]->(e2:Entity)
                    WITH d, c, e, e1, r, e2
                    LIMIT $limit
                    RETURN {
                        nodes: COLLECT(DISTINCT {
                            id: d.id, 
                            label: d.title, 
                            type: 'document'
                        }) + 
                        COLLECT(DISTINCT {
                            id: e.name, 
                            label: e.name, 
                            type: e.type
                        }),
                        links: COLLECT(DISTINCT {
                            source: d.id,
                            target: e.name,
                            type: "CONTAINS"
                        }) +
                        COLLECT(DISTINCT {
                            source: e1.name,
                            target: e2.name,
                            type: r.type
                        })
                    } as graph
                    """,
                    limit=limit
                )
                r = result.single()
                return r["graph"] if r else None
    except Exception as e:
        logger.error(f"图数据获取失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/api/entity-node")
def get_entity_node(name: str):
    try:
        if graph_rag and graph_rag.neo4j_manager:
             with graph_rag.neo4j_manager.driver.session() as session:
                result = session.run("""
                    MATCH (e:Entity)
                    WHERE toLower(e.name) CONTAINS toLower($name)
                    RETURN id(e) AS node_id, e.name AS name, e.type AS type, e.properties AS properties
                    ORDER BY size(e.name)
                    LIMIT 1
                """, name=name)
                record = result.single()
                if record:
                    return record.data()
                else:
                    return {"message": "Entity not found"}
    except Exception as e:
        logger.error(f"文本id获取失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
                       
@app.get("/api/graph/centered")
async def get_centered_graph(node_id: str):
    """获取以某节点为中心的子图"""
    try:
        if not graph_rag or not graph_rag.neo4j_manager:
            raise HTTPException(status_code=500, detail="系统未初始化")

        with graph_rag.neo4j_manager.driver.session() as session:
            result = session.run(
                """
                MATCH (n)
                WHERE n.id = $node_id OR n.name = $node_id

                // 获取1度邻居（出邻居）
                OPTIONAL MATCH (n)-[r1]->(m)
                // 获取1度邻居（入邻居）  
                OPTIONAL MATCH (p)-[r2]->(n)

                // 过滤掉null值
                WITH n, 
                    [node IN COLLECT(DISTINCT m) WHERE node IS NOT NULL] AS outNeighbors,
                    [node IN COLLECT(DISTINCT p) WHERE node IS NOT NULL] AS inNeighbors,
                    [rel IN COLLECT(DISTINCT r1) WHERE rel IS NOT NULL] AS outRels,
                    [rel IN COLLECT(DISTINCT r2) WHERE rel IS NOT NULL] AS inRels

                // 合并所有节点，中心节点排在前面
                WITH [n] + outNeighbors + inNeighbors AS allNodes,
                    outRels + inRels AS allRels

                // 限制节点数量为50
                WITH allNodes[0..50] AS limitedNodes, allRels

                // 过滤关系，只保留在限制节点范围内的关系
                WITH limitedNodes,
                    [rel IN allRels WHERE 
                    startNode(rel) IN limitedNodes AND 
                    endNode(rel) IN limitedNodes] AS validRels

                RETURN {
                    nodes: [node IN limitedNodes WHERE node IS NOT NULL | {
                        id: 
                            CASE 
                                WHEN node.id IS NOT NULL THEN node.id
                                ELSE node.name + "_" + COALESCE(node.source_doc, "")
                            END,
                        label: coalesce(node.title, node.name),
                        type: coalesce(node.type, labels(node)[0])
                    }],
                    links: [rel IN validRels WHERE rel IS NOT NULL | {
                        source: 
                            CASE 
                                WHEN startNode(rel).id IS NOT NULL THEN startNode(rel).id
                                ELSE startNode(rel).name + "_" + COALESCE(startNode(rel).source_doc, "")
                            END,
                        target: 
                            CASE 
                                WHEN endNode(rel).id IS NOT NULL THEN endNode(rel).id
                                ELSE endNode(rel).name + "_" + COALESCE(endNode(rel).source_doc, "")
                            END,
                        type: type(rel)
                    }]
                } AS graph

                """,
                node_id=node_id
            )
            record = result.single()
            return record["graph"] if record else {"nodes": [], "links": []}
    except Exception as e:
        logger.error(f"获取中心图失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/graph/3d-visualization")
async def get_3d_visualization_data(limit: int = 50):
    """获取3D可视化数据"""
    try:
        if graph_rag and graph_rag.neo4j_manager:
            with graph_rag.neo4j_manager.driver.session() as session:
                # 获取文档、实体和关系数据
                result = session.run(
                    """
                    MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
                    OPTIONAL MATCH (d)-[:CONTAINS_ENTITY]->(e:Entity)
                    OPTIONAL MATCH (e1:Entity)-[r:RELATED]->(e2:Entity)
                    WITH d, c, e, e1, r, e2
                    LIMIT $limit
                    RETURN {
                        nodes: COLLECT(DISTINCT {
                            id: d.id, 
                            name: d.title,
                            type: 'document',
                            size: 5,
                            color: 0x667eea
                        }) + 
                        COLLECT(DISTINCT {
                            id: e.name, 
                            name: e.name,
                            type: e.type,
                            size: 3,
                            color: CASE e.type
                                WHEN 'person' THEN 0xF5B7B1
                                WHEN 'location' THEN 0xA9DFBF
                                ELSE 0x764ba2
                            END
                        }),
                        links: COLLECT(DISTINCT {
                            source: d.id,
                            target: e.name,
                            strength: 0.5
                        }) +
                        COLLECT(DISTINCT {
                            source: e1.name,
                            target: e2.name,
                            strength: 0.8
                        })
                    } as graph
                    """,
                    limit=limit
                )
                r = result.single()
                return r["graph"] if r else None
    except Exception as e:
        logger.error(f"3D图数据获取失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    print("启动GraphRAG知识库系统...")
    print("请确保以下服务正在运行：")
    print("1. Ollama: ollama serve")
    print("2. Neo4j: docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest")
    print("\n系统将在 http://localhost:8000 启动")
    
    try:
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    except Exception as e:
        logger.error(f"启动失败: {e}")
        input("按Enter键退出...")  # 防止窗口立即关闭
 
        
        