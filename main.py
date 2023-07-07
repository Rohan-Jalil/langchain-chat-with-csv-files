from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from langchain.agents.agent_types import AgentType

agent = create_csv_agent(
    OpenAI(temperature=0),
    "./files_to_read/test.csv",
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

agent.run("How many totals rows are there in the dataset?")