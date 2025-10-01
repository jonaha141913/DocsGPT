from application.agents.classic_agent import ClassicAgent
from application.agents.react_agent import ReActAgent


class AgentCreator:
    agents = None

    def __init__(self):
        if AgentCreator.agents is None:
            AgentCreator.agents = {
                "classic": ClassicAgent,
                "react": ReActAgent,
            }

    @classmethod
    def create_agent(cls, type, *args, **kwargs):
        if cls.agents is None:
            cls.agents = {
                "classic": ClassicAgent,
                "react": ReActAgent,
            }
        agent_class = cls.agents.get(type.lower())
        if not agent_class:
            raise ValueError(f"No agent class found for type {type}")
        return agent_class(*args, **kwargs)
