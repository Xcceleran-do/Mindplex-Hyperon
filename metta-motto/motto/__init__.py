# needed for import! motto
# Load environment variables from .env file
import os
import importlib.util
try:
    from dotenv import load_dotenv
    # Look for .env file in the project root (parent of motto directory)
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        # Also try .environ file as fallback
        environ_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.environ')
        if os.path.exists(environ_path):
            load_dotenv(environ_path)
except ImportError:
    # python-dotenv not available, skip loading
    pass

from .llm_gate import llmgate_atoms, postproc_atoms
from .sparql_gate import sql_space_atoms
from .utils import get_string_value, get_token_from_stream_response, get_sentence_from_stream_response, get_ticks
from .langchain_agents.langchain_agent import langchaingate_atoms
from .thread_agents.thread_agents import listening_gate_atoms
from .snet_sdk_agents import snet_sdk_atoms
