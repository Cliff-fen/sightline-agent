from __future__ import annotations

TOOLS = [
    {"type": "function", "function": {"name": "web_search", "description": "Search ranked web evidence.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "topK": {"type": "integer"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "text_search", "description": "Search factual text evidence.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "topK": {"type": "integer"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "visit", "description": "Read a webpage.", "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
    {"type": "function", "function": {"name": "image_search", "description": "Find visually similar images.", "parameters": {"type": "object", "properties": {"image": {"type": "string"}}, "required": ["image"]}}},
    {"type": "function", "function": {"name": "crop", "description": "Crop a visual region.", "parameters": {"type": "object", "properties": {"image": {"type": "string"}, "x": {"type": "integer"}, "y": {"type": "integer"}, "width": {"type": "integer"}, "height": {"type": "integer"}}, "required": ["image", "x", "y", "width", "height"]}}},
    {"type": "function", "function": {"name": "layout_parsing", "description": "Extract text and reading order from an image.", "parameters": {"type": "object", "properties": {"image": {"type": "string"}}, "required": ["image"]}}},
    {"type": "function", "function": {"name": "sharpen", "description": "Enhance local image detail when the image is blurry.", "parameters": {"type": "object", "properties": {"image": {"type": "string"}, "amount": {"type": "number", "minimum": 0, "maximum": 5}}, "required": ["image"]}}},
    {"type": "function", "function": {"name": "super_resolution", "description": "Upscale a low-resolution image before inspection.", "parameters": {"type": "object", "properties": {"image": {"type": "string"}, "scale": {"type": "number", "minimum": 1.1, "maximum": 4}}, "required": ["image"]}}},
    {"type": "function", "function": {"name": "perspective_correct", "description": "Correct a mild perspective or rotation distortion.", "parameters": {"type": "object", "properties": {"image": {"type": "string"}, "angle": {"type": "number", "minimum": -20, "maximum": 20}}, "required": ["image"]}}},
]
