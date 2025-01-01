from typing import List, Optional, TypedDict

from langchain.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from langgraph.graph.graph import CompiledGraph

from gen_ui_backend.tools.github import github_repo
from gen_ui_backend.tools.invoice import invoice_parser
from gen_ui_backend.tools.weather import weather_data


class GenUIState(TypedDict, total=False):
    input: HumanMessage
    result: Optional[str]
    """Gives a plain text response if no tool was used."""
    tool_calls: Optional[List[dict]]
    """Gives a list of parsed tool calls."""
    tool_result: Optional[dict]
    """Gives the result of a tool call."""

def create_graph() -> CompiledGraph:
    builder = StateGraph(GenUIState)

    
    builder.add_node("invoke_model", invoke_model)
    builder.add_node("invoke_tools", invoke_tools)
    builder.add_conditional_edges("invoke_model", invoke_tools_or_return)
    
    builder.add_edge(START, "invoke_model")
    builder.add_edge("invoke_tools", END)

    #builder.set_entry_point("invoke_model")
    #builder.set_finish_point("invoke_tools")

    graph = builder.compile()
    return graph


def invoke_model(state: GenUIState, config: RunnableConfig) -> GenUIState:
    tools_parser = JsonOutputToolsParser()
    initial_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful all-knowing, AI assistant. You're provided a list of tools and an input from the user.\n"
                + "Your job is to determine whether or not you have a tool which can handle the users input, or respond with plain text.",
            ),
            MessagesPlaceholder("input"),
        ]
    )
    
    model = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)
    tools = [github_repo, invoice_parser, weather_data]
    model_with_tools = model.bind_tools(tools)
    chain = initial_prompt | model_with_tools
    result = chain.invoke({"input": state["input"]}, config)

    if not isinstance(result, AIMessage):
        raise ValueError("Resulted in an invalid result from the model. Expected AIMessage.")

    if isinstance(result.tool_calls, list) and len(result.tool_calls) > 0:
        parsed_tools = tools_parser.invoke(result, config)
        return {"tool_calls": parsed_tools}
    else:
        return {"result": str(result.content)}


def invoke_tools_or_return(state: GenUIState) -> str:
    if "result" in state and isinstance(state["result"], str):
        return END
    elif "tool_calls" in state and isinstance(state["tool_calls"], list):
        return "invoke_tools"
    else:
        raise ValueError("Resulted in an invalid state. There were no result or tool calls were found.")


def invoke_tools(state: GenUIState) -> GenUIState:
    tools_map = {
        "github-repo": github_repo,
        "invoice-parser": invoice_parser,
        "weather-data": weather_data,
    }

    if state["tool_calls"] is not None:
        tool = state["tool_calls"][0]
        selected_tool = tools_map[tool["type"]]
        return {"tool_result": selected_tool.invoke(tool["args"])}
    else:
        raise ValueError("There were no tool calls found in state.")
