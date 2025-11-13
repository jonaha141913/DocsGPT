from application.agents.classic_agent import ClassicAgent
from application.agents.react_agent import ReActAgent
import logging

logger = logging.getLogger(__name__)


class AgentCreator:
    agents = {
        "classic": ClassicAgent,
        "react": ReActAgent,
    }

    @classmethod
    def create_agent(cls, type, *args, **kwargs):
        agent_class = cls.agents.get(type.lower())
        if not agent_class:
            raise ValueError(f"No agent class found for type {type}")
        
        gpt_model = kwargs.pop('gpt_model', None)
        if gpt_model:
            kwargs['gpt_model'] = gpt_model
        
        return agent_class(*args, **kwargs)
