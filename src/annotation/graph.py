"""LangGraph StateGraph for cultural VLM annotation."""

from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.annotation.config import AnnotatorSettings
from src.annotation.nodes.annotator import make_annotator_node
from src.annotation.nodes.assembler import assembler_node
from src.annotation.nodes.image_loader import image_loader_node
from src.annotation.state import CulturalAnnotationState


def build_annotation_graph(
    settings: AnnotatorSettings,
) -> CompiledStateGraph[
    CulturalAnnotationState, None, CulturalAnnotationState, CulturalAnnotationState
]:
    """
    Build and compile the LangGraph pipeline:
      [image_loader] → [assembler] → [annotator] → END
    """
    annotator_node: Any = make_annotator_node(settings)

    graph = StateGraph(CulturalAnnotationState)
    graph.add_node("image_loader", image_loader_node)
    graph.add_node("assembler", assembler_node)
    graph.add_node("annotator", annotator_node)

    graph.set_entry_point("image_loader")
    graph.add_edge("image_loader", "assembler")
    graph.add_edge("assembler", "annotator")
    graph.add_edge("annotator", END)

    return graph.compile()
