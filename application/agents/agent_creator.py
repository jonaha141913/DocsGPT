from application.agents.classic_agent import ClassicAgent
from application.agents.react_agent import ReActAgent


class AgentCreator:
    agents: ClassVar[dict[str, type]] = {}

    @classmethod
    def _get_agents(cls) -> dict[str, type]:
        if not cls.agents:
            cls.agents = {
                "classic": ClassicAgent,
                "react": ReActAgent,
            }
        return cls.agents

    @classmethod
    def create_agent(cls, type, *args, **kwargs):
        agent_class = cls._get_agents().get(type.lower())
        if not agent_class:
            raise ValueError(f"No agent class found for type {type}")
        return agent_class(*args, **kwargs)
