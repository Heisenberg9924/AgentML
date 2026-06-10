from config.llm import llm
from agents.data_agent import DataAgent
from agents.feature_agent import FeatureAgent
from agents.experiment_agent import ExperimentAgent
from agents.critic_agent import CriticAgent


def data_node(state):

    agent = DataAgent(llm)

    return agent.run(state)


def feature_node(state):

    agent = FeatureAgent(llm)

    return agent.run(state)


def experiment_node(state):

    agent = ExperimentAgent()

    return agent.run(state)


def critic_node(state):

    agent = CriticAgent(llm)

    return agent.run(state)