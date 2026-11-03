from .h3_semantic.nodes import H3SemanticTimeline, H3SemanticTimelinePreview, H3SemanticTimelineEncode
from .h3_semantic.version import __version__
NODE_CLASS_MAPPINGS = {
    'H3SemanticTimeline': H3SemanticTimeline,
    'H3SemanticTimelinePreview': H3SemanticTimelinePreview,
    'H3SemanticTimelineEncode': H3SemanticTimelineEncode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    'H3SemanticTimeline': 'H3 Semantic Timeline',
    'H3SemanticTimelinePreview': 'H3 Semantic Timeline Preview',
    'H3SemanticTimelineEncode': 'H3 Semantic Timeline Encode',
}

WEB_DIRECTORY = './web'
