from application.agents.classic_agent import ClassicAgent
from application.agents.react_agent import ReActAgent


class AgentCreator:
    @classmethod
    def create_agent(cls, type, *args, **kwargs):
        agents = {
            "classic": ClassicAgent,
            "react": ReActAgent,
        }
        agent_class = agents.get(type.lower())
        if not agent_class:
            raise ValueError(f"No agent class found for type {type}")
        return agent_class(*args, **kwargs)
