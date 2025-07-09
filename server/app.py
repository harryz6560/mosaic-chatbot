import logging
from typing import Annotated, Any
from fastapi import Depends, FastAPI
from pydantic import BaseModel, ValidationError
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import traceback
from fastapi.responses import JSONResponse
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    ChatMessage,
    ChatMessageRole,
)
from dotenv import load_dotenv
from fastapi import Request
from fastapi.responses import StreamingResponse
import asyncio
from databricks import sql
from functools import lru_cache
import re
from databricks_langchain import DatabricksEmbeddings
from langchain_community.vectorstores import DatabricksVectorSearch
from databricks.vector_search.client import VectorSearchClient
import json
import time
from fastapi import UploadFile, File
import pandas as pd
import PyPDF2
import io


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.info("Logger initialized successfully!")

load_dotenv()

ENDPOINT_NAME = os.getenv("SERVING_ENDPOINT_NAME")

SILVER_METADATA = None

SYSTEM_PROMPT = """
You are an assistant and an expert on Oil and Gas Unconventioal projects that can reason and use tools.

You have access to these tools:
1. doc_tool(question: str) — returns geology/O&G document context
2. sql_tool(question: str) — returns relevant database table metadata
3. run_sql(query: str) — executes SQL queries and returns results

If a question requires using one of these tools, respond ONLY with a JSON block like:
{"tool": "doc_tool", "input": {"question": "What is porosity of Montney?"}}

After receiving tool results, use them to reason. Avoid repeating tool calls if the result seems valid or helpful.

Never include explanation with the tool call — only JSON.

Once you have sufficient information, respond with your final answer in English.
"""


TOOL_DEFINITIONS = [
    {
        "name": "doc_tool",
        "description": "Fetch geology/O&G document context.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The user question."}
            },
            "required": ["question"],
        },
    },
    {
        "name": "sql_tool",
        "description": "Fetch metadata about relevant database tables.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The user question."}
            },
            "required": ["question"],
        },
    },
    {
        "name": "run_sql",
        "description": "Execute a SQL query and return results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A valid SQL query."}
            },
            "required": ["query"],
        },
    },
]


if not ENDPOINT_NAME:
    logger.error("SERVING_ENDPOINT_NAME environment variable is not set")
    raise ValueError("SERVING_ENDPOINT_NAME environment variable is not set")

app = FastAPI()
ui_app = StaticFiles(directory="client/build", html=True)
api_app = FastAPI()

# PLEASE NOTE THE ORDER OF THE MOUNTS MATTERS
app.mount("/api", api_app)
app.mount("/", ui_app)

origins = [
    "http://localhost:3000",
]

# Make sure CORS is applied to both app and api_app
# This is only needed for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# client
def client():
    try:
        logger.info("Creating WorkspaceClient...")
        c = WorkspaceClient()
        logger.info(f"WorkspaceClient created with host: {c.config.host}")
        return c
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"Failed to create WorkspaceClient: {e}")
        logger.error(tb)
        raise


# Model for the request body
class ChatMessageInput(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessageInput] = []

# Simplified response model
class ChatResponse(BaseModel):
    content: str

"""
@lru_cache(maxsize=1)
def get_cached_metadata():
    global SILVER_METADATA
    logger.info("Generating silver metadata...")
    if SILVER_METADATA is None:
        SILVER_METADATA = generate_table_metadata()
    return SILVER_METADATA
"""

@lru_cache(maxsize=1)
def get_database_retriever():
    logger.info("Loading vector retriever for table metadata...")
    embedding_model = DatabricksEmbeddings(endpoint="databricks-gte-large-en")
    vsc = VectorSearchClient()
    vs_index = vsc.get_index(
        endpoint_name="database_vector_endpoint",
        index_name="analytics.llm.database_vector_index"
    )
    vectorstore = DatabricksVectorSearch(
        vs_index,
        embedding=embedding_model,
        text_column="combined_text",
        columns=["full_table_name", "description", "columns"]
    )
    return vectorstore.as_retriever(search_kwargs={"k": 10})

@lru_cache(maxsize=1)
def get_doc_retriever():
    logger.info("Loading vector retriever for table metadata...")
    embedding_model = DatabricksEmbeddings(endpoint="databricks-gte-large-en")
    vsc = VectorSearchClient()
    vs_index = vsc.get_index(
        endpoint_name="doc_vector_endpoint",
        index_name="analytics.llm.docs_idx"
    )
    vectorstore = DatabricksVectorSearch(
        vs_index,
        embedding=embedding_model,
        text_column="text",
        columns=["file_name"]
    )
    return vectorstore.as_retriever(search_kwargs={"k": 10})

def doc_tool(question: str) -> str:
    docs = get_doc_retriever().get_relevant_documents(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    return context

def sql_tool(question: str) -> str:
    return get_table_metadata_context(question) or ""

def get_table_metadata_context(question: str) -> str:
    retriever = get_database_retriever()
    docs = retriever.get_relevant_documents(question)
    return "\n\n".join(doc.page_content for doc in docs if doc.page_content.strip())

"""
@lru_cache(maxsize=1)
def get_cached_metadata():
    global SILVER_METADATA
    metadata_file = os.path.join(os.path.dirname(__file__), "silver_metadata.txt")

    if not os.path.exists(metadata_file):
        logger.error(f"{metadata_file} not found. Please run generate_metadata.py before starting the server.")
        return ""
    if SILVER_METADATA == None:
        logger.info(f"Loading cached metadata from {metadata_file}")
        with open(metadata_file, "r", encoding="utf-8") as f:
            SILVER_METADATA = f.read()
    return SILVER_METADATA
"""

def generate_table_metadata() -> str:
    conn = sql.connect(
        server_hostname=os.getenv("DATABRICKS_HOST"),
        http_path=os.getenv("DATABRICKS_SQL_HTTP_PATH"),
        access_token=os.getenv("SQL_ACCESS_TOKEN"),
        _user_agent_entry="chatbot-v1",  # Optional, helps with logs
        # Force use of provided config only (no ~/.databrickscfg, no OAuth fallback)
        _use_global_config=False
    )
    cursor = conn.cursor()

    cursor.execute("SHOW SCHEMAS IN silver")
    schemas = [row[0] for row in cursor.fetchall()]

    metadata = []
    for schema in schemas:
        cursor.execute(f"SHOW TABLES IN silver.{schema}")
        tables = [row[1] for row in cursor.fetchall()]

        for table in tables:
            cursor.execute(f"DESCRIBE TABLE silver.{schema}.{table}")
            columns = cursor.fetchall()
            col_list = ", ".join(f"{col[0]} ({col[1]})" for col in columns)
            metadata.append(f"Table: {schema}.{table}\nColumns: {col_list}\n")

    conn.close()
    return "\n".join(metadata)

def is_sql_query(text: str) -> bool:
    # Look for a SELECT ... FROM pattern anywhere in the text
    return bool(re.search(r"\bselect\b.+?\bfrom\b", text, re.IGNORECASE | re.DOTALL))

def extract_sql(text: str) -> str:
    # Try to extract SQL block from markdown-style response
    match = re.search(r"```sql\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Fallback: try to extract first SELECT ... FROM ... statement
    match = re.search(r"(select\b.*?\bfrom\b.*?);", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback to raw text
    return text.strip()

def run_sql(query: str) -> str:
    logger.info(f"Running SQL query: {query}")

    """
    # Safeguard: if no LIMIT, add one (unless it's an aggregation)
    if "limit" not in query.lower() and not any(kw in query.lower() for kw in ["count", "sum", "avg", "max", "min"]):
        query += " LIMIT 100"
    """

    start = time.time()
    conn = sql.connect(
        server_hostname=os.getenv("DATABRICKS_HOST"),
        http_path=os.getenv("DATABRICKS_SQL_HTTP_PATH"),
        access_token=os.getenv("SQL_ACCESS_TOKEN"),
        _user_agent_entry="chatbot-v1",
        _use_global_config=False
    )
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    conn.close()
    elapsed = round(time.time() - start, 2)
    logger.info(f"SQL completed in {elapsed}s and returned {len(rows)} rows")

    if not rows:
        return "No data found."

    # Truncate long cells
    def truncate(cell, maxlen=80):
        return str(cell)[:maxlen] + "…" if len(str(cell)) > maxlen else str(cell)

    # Calculate width
    col_widths = [min(max(len(str(col)), 10), 30) for col in columns]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = min(max(col_widths[i], len(str(cell))), 30)

    def format_row(row):
        return " | ".join(truncate(cell).ljust(col_widths[i]) for i, cell in enumerate(row))

    """
    header = format_row(columns)
    separator = "-+-".join("-" * width for width in col_widths)
    data = "\n".join(format_row(row) for row in rows)
    """
    df = pd.DataFrame(rows, columns=columns)

    return df.to_markdown(index=False)


def extract_tool_json(text: str) -> dict | None:
    try:
        match = re.search(r'{\s*"tool"\s*:\s*".+?"\s*,\s*"input"\s*:\s*{.*}}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception as e:
        logger.warning(f"Failed to parse tool JSON: {e}")
    return None


@api_app.post("/deep_think", response_model=ChatResponse)
async def deep_think(request: Request, client: Annotated[WorkspaceClient, Depends(client)]):
    try:
        body = await request.json()
        req = ChatRequest(**body)

        messages = [
            ChatMessage(role=ChatMessageRole.SYSTEM, content=SYSTEM_PROMPT)
        ]
        for msg in req.history:
            messages.append(ChatMessage(content=msg.content, role=ChatMessageRole(msg.role)))
        messages.append(ChatMessage(content=req.message, role=ChatMessageRole.USER))

        tool_call_memory = set()
        max_loop = 5

        for i in range(max_loop):
            logger.info(f"Invoking Claude - iteration {i+1}")
            response = client.serving_endpoints.query(
                name=ENDPOINT_NAME,
                messages=messages,
            )

            msg = response.choices[0].message
            logger.info(f"Claude reply: {msg.content}")

            tool_call = extract_tool_json(msg.content)

            if tool_call:
                tool_name = tool_call["tool"]
                args = tool_call["input"]

                # Deduplicate tool calls
                tool_key = json.dumps(tool_call, sort_keys=True)
                if tool_key in tool_call_memory:
                    logger.info(f"Duplicate tool call detected (skipped): {tool_key}")
                    messages.append(ChatMessage(
                        role=ChatMessageRole.USER,
                        content="You already tried that tool call. Please consider a different approach."
                    ))
                    continue
                tool_call_memory.add(tool_key)

                is_tool_error = False

                try:
                    result = globals()[tool_name](**args)
                except Exception as tool_err:
                    tb = traceback.format_exc()
                    logger.error(f"Tool `{tool_name}` failed:\n{tb}")
                    result = f"[TOOL ERROR] `{tool_name}` failed with error:\n{tool_err}"
                    is_tool_error = True

                messages.append(ChatMessage(
                    role=ChatMessageRole.USER,
                    content=f"Tool `{tool_name}` executed with input:\n{json.dumps(args, indent=2)}\n\nResult:\n{result}"
                ))

                # Add assistant hint based on tool result
                if tool_name == "run_sql":
                    if is_tool_error:
                        messages.append(ChatMessage(
                            role=ChatMessageRole.USER,
                            content="The SQL query failed. Please revise your approach."
                        ))
                    elif "No data found" in result:
                        messages.append(ChatMessage(
                            role=ChatMessageRole.USER,
                            content="The previous SQL query returned no results."
                        ))
                    else:
                        messages.append(ChatMessage(
                            role=ChatMessageRole.ASSISTANT,
                            content="Thanks, I now have the data I need to answer your question."
                        ))

                continue  # loop again with new context

            else:
                logger.info("Returning final answer from Claude.")
                return ChatResponse(content=msg.content)

        logger.warning("Reached max tool-call loop limit.")
        return ChatResponse(
            content="Loop limit reached without resolving your query. Please refine your question or try again."
        )

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Exception in /deep_think:\n{tb}")
        return JSONResponse(status_code=500, content={"detail": str(e), "traceback": tb})
    
    
@api_app.post("/chat", response_model=ChatResponse)
async def normal_chat(
    request: Request, client: Annotated[WorkspaceClient, Depends(client)]
):
    try:
        json_body = await request.json()
        logger.info(f"Parsed JSON: {json_body}")

        request = ChatRequest(**json_body)

        logger.info(f"Received message: {request.message}")
        logger.info(f"Received history: {request.history}")
        logger.info(f"Using endpoint: {ENDPOINT_NAME}")

        # Before calling Claude:
        full_data = "\n".join([msg.content for msg in request.history] + [request.message])

        # Step 1: Retrieve from vector DBs
        doc_context = ""
        sql_context = ""
        try:
            doc_context = doc_tool(full_data)
            sql_context = get_table_metadata_context(full_data)
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Exception in /chat:\n{tb}")


        system_message = f"""
        You are an assistant for Geology Knowledge Related to Oil and Gas Unconventional projects and users.

        You are answering how-to, understanding, and inference questions regarding the Geosciences and Petroleum Engineering you have data on.
        If you don't know the answer, just say that you don't know. Do not fabricate an answer.
        If the question appears to be about an oil and gas asset or field that you don't have data on, say so.

        Keep the answer as concise as possible. Provide all answers only in English.

        If the user is asking a question that requires data lookup, determine:
        1. Which table from the silver.<schema>.<table> catalog is most relevant.
        2. Which fields (columns) are relevant.
        3. Write a valid SQL query to retrieve the answer. Use fully qualified names like silver.schema.table.

        ### Document Context ###
        {doc_context}

        ### Available Metadata ###
        Below is the schema/table/column metadata you may query:

        {sql_context}

        ### Response Instructions ###
        - If the question does not require a SQL query, reply naturally.
        - If it does, include a valid SQL query in markdown format like ```sql ... ```
        """.strip()


        # Convert chat history into SDK-compatible ChatMessages
        messages = [ChatMessage(role=ChatMessageRole.SYSTEM, content=system_message)]

        for msg in request.history:
            messages.append(ChatMessage(content=msg.content, role=ChatMessageRole(msg.role)))


        # Append current user message
        messages.append(
            ChatMessage(content=request.message, role=ChatMessageRole.USER)
        )

        response = client.serving_endpoints.query(
            ENDPOINT_NAME,
            messages=messages,
        )

        claude_reply = response.choices[0].message.content.strip()
        logger.info(f"Claude reply: {claude_reply}")

        # New logic — check if Claude returned SQL
        if is_sql_query(claude_reply):
            try:
                sql_to_run = extract_sql(claude_reply)
                result = run_sql(sql_to_run)
                return ChatResponse(content=result)
            except Exception as db_err:
                tb = traceback.format_exc()
                logger.error(f"Failed to run SQL: {db_err}\n{tb}")
                return ChatResponse(content=f"SQL query failed:\n{str(db_err)}")
        
        return ChatResponse(content=claude_reply)

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Exception during chat_with_llm:\n{tb}")
        return JSONResponse(
            status_code=500,
            content={"detail": str(e), "traceback": tb}
        )

@api_app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename.lower()
    content = ""

    try:
        if filename.endswith(".pdf"):
            raw = await file.read()
            pdf = PyPDF2.PdfReader(io.BytesIO(raw))
            content = "\n".join(page.extract_text() or "" for page in pdf.pages)

        elif filename.endswith(".txt"):
            content = (await file.read()).decode("utf-8")

        elif filename.endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(await file.read()), engine="python", on_bad_lines="skip")
                content = df.head(10).to_string(index=False)
            except Exception as e:
                content = f"Failed to parse CSV: {str(e)}"

        elif filename.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(await file.read()))
            content = df.to_string(index=False)

        else:
            content = "Unsupported file type."

    except Exception as e:
        return {"content": f"Failed to parse file: {str(e)}"}

    # Optional truncation to prevent overwhelming chat
    return {"content": content[:3000]}


@api_app.get("/list-endpoints")
def list_endpoints(client: Annotated[WorkspaceClient, Depends(client)]):
    eps = client.serving_endpoints.list()
    return {"endpoints": [e.name for e in eps]}


@api_app.post("/chat-stream")
async def chat_stream(request: Request, client = Depends(client)):
    body = await request.json()
    parsed = ChatRequest(**body)

    messages = [
        ChatMessage(content=msg.content, role=ChatMessageRole(msg.role))
        for msg in parsed.history
    ] + [ChatMessage(content=parsed.message, role=ChatMessageRole.USER)]

    # Stream tokens directly from Databricks if supported
    response = client.serving_endpoints.query(
        ENDPOINT_NAME,
        messages=messages,
        stream=True  # enable streaming
    )

    async def event_stream():
        async for chunk in response:
            yield chunk.message.content

    return StreamingResponse(event_stream(), media_type="text/plain")