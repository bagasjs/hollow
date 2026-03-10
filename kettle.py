#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Kettle is a simple micro template engine all in a single file 
and with no dependencies other than the Python Standard Library.

Kettle only supports
- If statement (no expression evaluation yet)
- For loop (without else if and else)

Copyright (c) 2025, bagasjs
License: MIT (see the details at the very bottom)
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional

TokenText = "TokenText"
TokenVar  = "TokenVar"
TokenIf   = "TokenIf"
TokenElse = "TokenElse"
TokenFor  = "TokenFor"
TokenEnd  = "TokenEnd"

class Token(object):
    kind: str
    value: str

    def __init__(self, kind: str, value: str, row: int):
        self.kind  = kind
        self.value = value
        self.row = row

    def __repr__(self):
        return "Token(kind=%s)" % (self.kind,)

AstNodeText = "AstNodeText"
AstNodeVar  = "AstNodeVar"
AstNodeIf   = "AstNodeIf"
AstNodeFor  = "AstNodeFor"
AstNodeBlock = "AstNodeBlock"

class AstNode:
    kind: str
    value: str
    body: List[AstNode]
    else_node: Optional[AstNode]
    row: int

    def __init__(self, kind: str, value: str, row: int):
        self.kind = kind
        self.value = value
        self.body = []
        self.row = row
        self.else_node = None

    # TODO: Recursion unrolling
    def render(self, data: Dict[str, Any]) -> str:
        if self.kind == AstNodeText:
            return self.value
        elif self.kind == AstNodeVar:
            if "." not in self.value:
                if self.value not in data:
                    raise ValueError(f"Could not find variable \"{self.value}\"")
                return data[self.value]
            else:
                parts = self.value.split(".")
                cursor: Any = data
                for i, part in enumerate(parts):
                    if part not in cursor:
                        raise ValueError(f"Could not find variable \"{'.'.join(parts[:i+1])}\"")
                    cursor = cursor[part]
                return cursor
        elif self.kind == AstNodeFor:
            iter = [ part for part in self.value.split(" ") if len(part) > 0 ]
            if len(iter) != 3 and iter[1] != "in":
                raise ValueError(f"Invalid for loop in row {self.row} -> {self.value}")
            if iter[2] not in data:
                raise ValueError(f"Could not find variable to iterate with name {iter[2]} at row {self.row}")
            body = []
            data[iter[0]] = None
            for item in data[iter[2]]:
                data[iter[0]] = item
                for part in self.body:
                    body.append(part.render(data))
            data.pop(iter[0])
            return "".join(body)
        elif self.kind == AstNodeBlock:
            return "".join([ child.render(data) for child in self.body ])

        # TODO: support expression based condition like a == b or a < b
        elif self.kind == AstNodeIf:
            iter = [ part for part in self.value.split(" ") if len(part) > 0 ]
            if data[self.value]:
                return "".join([ child.render(data) for child in self.body ])
            elif self.else_node is not None:
                return "".join([ child.render(data) for child in self.else_node.body ])
        return ""

class Template(object):
    _program: Optional[AstNode]
    _source: str

    def __init__(self, source: str):
        self._source = source
        self._program = None
        self._prepare()

    def _tokenize(self) -> List[Token]:
        tokens = []
        i = 0
        text = ""
        row = 0
        while i < len(self._source):
            if self._source[i] == "\n":
                row += 1

            if self._source[i:].startswith("{{"):
                if len(text) > 0:
                    tokens.append(Token(kind=TokenText, value=text, row=row))
                    text = ""
                end = self._source[i:].find("}}")
                tokens.append(Token(kind=TokenVar, value=self._source[i+2:i+end], row=row))
                i += end + 2
            elif self._source[i:].startswith("{%"):
                if len(text) > 0:
                    tokens.append(Token(kind=TokenText, value=text, row=row))
                    text = ""
                end = self._source[i:].find("%}")
                value = self._source[i+2:i+end].strip()
                i += end + 2

                if value.startswith("for "):
                    tokens.append(Token(kind=TokenFor, value=value[4:], row=row))
                elif value.startswith("if "):
                    tokens.append(Token(kind=TokenIf, value=value[3:], row=row))
                elif value == "else":
                    tokens.append(Token(kind=TokenElse, value=value[3:], row=row))
                elif value == "end":
                    tokens.append(Token(kind=TokenEnd, value=value, row=row))
                else:
                    raise ValueError(f"Invalid statement \"{value}\" at row {row+1}")
            else:
                text += self._source[i]
                i += 1

        if len(text) > 0:
            tokens.append(Token(kind=TokenText, value=text, row=row))
        return tokens

    def _prepare(self):
        self._program = AstNode(AstNodeBlock, "BLOCK", 0)
        tokens = self._tokenize()
        i = 0
        scope = [ self._program ]
        while i < len(tokens):
            if tokens[i].kind == TokenText:
                scope[-1].body.append(AstNode(AstNodeText, tokens[i].value, tokens[i].row))
            elif tokens[i].kind == TokenVar:
                scope[-1].body.append(AstNode(AstNodeVar, tokens[i].value, tokens[i].row))
            elif tokens[i].kind == TokenFor:
                node = AstNode(AstNodeFor, tokens[i].value, tokens[i].row)
                scope[-1].body.append(node)
                scope.append(node)
            elif tokens[i].kind == TokenIf:
                node = AstNode(AstNodeIf, tokens[i].value, tokens[i].row)
                scope[-1].body.append(node)
                scope.append(node)
            elif tokens[i].kind == TokenElse:
                if scope[-1].kind != AstNodeIf:
                    raise ValueError(f"Invalid usage of else at row {tokens[i].row} at {scope[-1].kind} block")
                else_node = AstNode(AstNodeBlock, "ELSE", tokens[i].row)
                scope[-1].else_node = else_node
                scope.append(else_node)
            elif tokens[i].kind == TokenEnd:
                node = scope.pop()
                if node.kind == AstNodeBlock and node.value == "ELSE":
                    if_node = scope.pop()
                    assert if_node.kind == AstNodeIf
            else:
                raise ValueError(f"Unknown token kind \"{tokens[i].kind}\"")
            i += 1

    def render(self, data: Dict[str, Any]) -> str:
        if self._program is None:
            self._prepare()
        return self._program.render(data)

if __name__ == "__main__":
    PAGE_HTML = """
    {% if login %}
        <p>Welcome, {{name}}!</p>
    {% else %}
        <p>Welcome, guest!</p>
    {% end %}

    <p>Products:</p>
    <ul>
    {% for product in products %}
        <li>{{product.name}} {{product.price}}</li>
    {% end %}

    </ul>
    """

    tpl = Template(PAGE_HTML)
    print("Rendered: ", tpl.render({
        "name": "Bagas",
        "login": True,
        "products": [
            {
                "name": "A",
                "price": "$1",
            },
            {
                "name": "B",
                "price": "$20",
            }
        ],
    }))
