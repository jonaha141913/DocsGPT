from application.agents.classic_agent import ClassicAgent
from application.agents.react_agent import ReActAgent


class AgentCreator:
    agents: dict[str, type] = {}

    @classmethod
    def create_agent(cls, type, *args, **kwargs):
        agent_class = cls.agents.get(type.lower())
        if not agent_class:
            raise ValueError(f"No agent class found for type {type}")
        return agent_class(*args, **kwargs)

    @classmethod
    def register_agent(cls, name: str, agent_class: type):
        cls.agents[name.lower()] = agent_class
