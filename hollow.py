from __future__ import annotations
from abc import ABC, abstractmethod
import requests
import os
import inspect

###########################################################
#
# Dependencies
#
###########################################################

# ===============================
# The Simple HTML Parser with DOM
# ===============================
"""
Bowl is a simple micro library for parsing HTML all in a single file 
and with no dependencies other than the Python Standard Library.

Copyright (c) 2025, bagasjs
License: MIT (see the details at the very bottom)
"""
from typing import List, Dict, Any, Optional, Union
from html.parser import HTMLParser
from html.entities import name2codepoint
# TODOs
# - in DOMNode at least in DOMNode._build_tag_index it will include itself into the indexes. This should not be happening
#   Although we don't need to remove it in DOMDocument since we might need it

class DOMNode(object):
    tag: str
    attrs: Dict[str, Any]
    children: List[Union[DOMNode, str]]

    # NOTE: This will be duplicating some indexes from DOMDocument and since an ID is unique per element there's no need to have this
    _id_to_node_map: Dict[str, DOMNode]
    _tag_to_node_map: Dict[str, List[DOMNode]]
    _class_to_node_map: Dict[str, List[DOMNode]]
    _has_id_index: bool
    _has_tag_index: bool
    _has_class_index: bool

    def __init__(self, tag: str, attrs: Dict[str, Any], children: Optional[List[Union[DOMNode, str]]] = None):
        self.tag = tag
        self.attrs = attrs
        self.children = children or []

        self._has_id_index = False
        self._id_to_node_map  = {}

        self._has_tag_index = False
        self._tag_to_node_map = {}

        self._has_class_index = False
        self._class_to_node_map = {}

    def append_child(self, child: Union[DOMNode, str]):
        self.children.append(child)

    def _render(self, indent: int = 0) -> str:
        space = "  " * indent
        attrs = " ".join(f'{key}="{value}"' for key, value in self.attrs.items())
        open_tag = f"<{self.tag}{' ' + attrs if attrs else ''}>"
        close_tag = f"</{self.tag}>"
        parts = [f"{space}{open_tag}"]

        for child in self.children:
            if isinstance(child, DOMNode):
                parts.append(child._render(indent + 1))
            elif isinstance(child, str):
                parts.append(f"{'  ' * (indent + 1)}{child}")

        parts.append(f"{space}{close_tag}")
        return "\n".join(parts)

    def __str__(self) -> str:
        return self._render()

    def __repr__(self) -> str:
        attrs = [ f"{key}=\"{value}\"" for key, value in self.attrs.items() ]
        if len(attrs) > 0:
            attr_as_str = " ".join(attrs)
            return f"<{self.tag} {attr_as_str}>"
        else:
            return f"<{self.tag}>"

    # TODO: since this is recursion maybe recursion unwrapping would be nice
    #       Look at https://github.com/bagasjs/algo-impl/blob/main/traversal.py
    def _build_tag_index(self, node: DOMNode):
        if node.tag not in self._tag_to_node_map:
            self._tag_to_node_map[node.tag] = []
        self._tag_to_node_map[node.tag].append(node)
        for child in node.children:
            if isinstance(child, DOMNode):
                self._build_tag_index(child)
        if node is self:
            self._has_tag_index = True

    # TODO: since this is recursion maybe recursion unwrapping would be nice
    #       Look at https://github.com/bagasjs/algo-impl/blob/main/traversal.py
    def _build_id_index(self, node: DOMNode):
        id_attr = node.attrs.get("id")
        if id_attr:
            self._id_to_node_map[id_attr] = node
        for child in node.children:
            if isinstance(child, DOMNode):
                self._build_id_index(child)
        if node is self:
            self._has_id_index = True

    # TODO: since this is recursion maybe recursion unwrapping would be nice
    #       Look at https://github.com/bagasjs/algo-impl/blob/main/traversal.py
    def _build_class_index(self, node: DOMNode):
        class_attr = node.attrs.get("class")
        if class_attr:
            classes = []
            if isinstance(class_attr, str):
                classes.extend(class_attr.split())
            elif isinstance(class_attr, list):
                classes.extend(class_attr)
            else:
                classes.append(str(class_attr))
            for class_name in classes:
                if class_name not in self._class_to_node_map:
                    self._class_to_node_map[class_name] = []
                self._class_to_node_map[class_name].append(node)

        for child in node.children:
            if isinstance(child, DOMNode):
                self._build_class_index(child)

        if node is self:
            self._has_class_index = True

    def inner_text(self) -> str:
        return "".join([str(child) for child in self.children])

    def get_by_tag(self, name: str) -> Optional[List[DOMNode]]:
        if not self._has_tag_index:
            self._build_tag_index(self)
        return self._tag_to_node_map.get(name)

    def get_by_id(self, name: str) -> Optional[DOMNode]:
        if not self._has_id_index:
            self._build_id_index(self)
        return self._id_to_node_map.get(name)

    def get_by_class_name(self, name: str) -> Optional[List[DOMNode]]:
        if not self._has_class_index:
            self._build_class_index(self)
        return self._class_to_node_map.get(name)

class DOMDocument(object):
    root: DOMNode
    page_title: Union[str, None]

    _id_to_node_map: Dict[str, DOMNode]
    _tag_to_node_map: Dict[str, List[DOMNode]]
    _class_to_node_map: Dict[str, List[DOMNode]]
    _has_id_index: bool
    _has_tag_index: bool
    _has_class_index: bool

    def __init__(self, root: DOMNode, page_title: Union[str, None]):
        self.root = root
        self.page_title = page_title
        self._has_id_index = False
        self._id_to_node_map  = {}

        self._has_tag_index = False
        self._tag_to_node_map = {}

        self._has_class_index = False
        self._class_to_node_map = {}

    # TODO: since this is recursion maybe recursion unwrapping would be nice
    #       Look at https://github.com/bagasjs/algo-impl/blob/main/traversal.py
    def _build_tag_index(self, node: DOMNode):
        if node.tag not in self._tag_to_node_map:
            self._tag_to_node_map[node.tag] = []
        self._tag_to_node_map[node.tag].append(node)
        for child in node.children:
            if isinstance(child, DOMNode):
                self._build_tag_index(child)
        if node is self.root:
            self._has_tag_index = True

    # TODO: since this is recursion maybe recursion unwrapping would be nice
    #       Look at https://github.com/bagasjs/algo-impl/blob/main/traversal.py
    def _build_id_index(self, node: DOMNode):
        id_attr = node.attrs.get("id")
        if id_attr:
            self._id_to_node_map[id_attr] = node
        for child in node.children:
            if isinstance(child, DOMNode):
                self._build_id_index(child)
        if node is self.root:
            self._has_id_index = True

    # TODO: since this is recursion maybe recursion unwrapping would be nice
    #       Look at https://github.com/bagasjs/algo-impl/blob/main/traversal.py
    def _build_class_index(self, node: DOMNode):
        class_attr = node.attrs.get("class")
        if class_attr:
            classes = []
            if isinstance(class_attr, str):
                classes.extend(class_attr.split())
            elif isinstance(class_attr, list):
                classes.extend(class_attr)
            else:
                classes.append(str(class_attr))
            for class_name in classes:
                if class_name not in self._class_to_node_map:
                    self._class_to_node_map[class_name] = []
                self._class_to_node_map[class_name].append(node)

        for child in node.children:
            if isinstance(child, DOMNode):
                self._build_class_index(child)

        if node is self.root:
            self._has_class_index = True

    def get_by_tag(self, name: str) -> Optional[List[DOMNode]]:
        if not self._has_tag_index:
            self._build_tag_index(self.root)
        return self._tag_to_node_map.get(name)

    def get_by_id(self, name: str) -> Optional[DOMNode]:
        if not self._has_id_index:
            self._build_id_index(self.root)
        return self._id_to_node_map.get(name)

    def get_by_class_name(self, name: str) -> Optional[List[DOMNode]]:
        if not self._has_class_index:
            self._build_class_index(self.root)
        return self._class_to_node_map.get(name)

VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", 
    "input", "link", "meta", "source", "track", "wbr"
}

class _CustomHTMLParser(HTMLParser):
    def __init__(self, source: str):
        super().__init__()
        self.source = source
        self.root_node = DOMNode("DOCUMENT_ROOT", {})
        self.title = None
        self.node_stack = [ self.root_node ]

    def handle_starttag(self, tag, attrs):
        dom_attrs = {}
        for key, value in attrs:
            dom_attrs[key] =  value
        dom = DOMNode(tag, dom_attrs)
        self.node_stack[-1].append_child(dom)
        if tag not in VOID_TAGS:
            self.node_stack.append(dom)

    def handle_endtag(self, tag):
        if tag == self.node_stack[-1].tag:
            dom = self.node_stack.pop()
            # print("Removed: ", dom.__repr__())

    def handle_data(self, data):
        if len(self.node_stack) > 0:
            if self.node_stack[-1].tag == "title" and self.title is None:
                self.title = data
            self.node_stack[-1].append_child(data)

    def handle_comment(self, data):
        # print("Comment  :", data)
        pass

    def handle_entityref(self, name):
        # c = chr(name2codepoint[name])
        # print("Named ent:", c)
        pass

    def handle_charref(self, name):
        # if name.startswith('x'):
        #     c = chr(int(name[1:], 16))
        # else:
        #     c = chr(int(name))
        # print("Num ent  :", c)
        pass

    def handle_decl(self, *args, **kwargs):
        # print("Decl     :", args, kwargs)
        pass

    def accumulate(self) -> Optional[DOMNode]:
        self.feed(self.source)
        assert self.root_node == self.node_stack[0]
        return self.root_node

def parse_document(source: str) -> Optional[DOMDocument]:
    parser = _CustomHTMLParser(source)
    root = parser.accumulate()
    if root:
        return DOMDocument(root, parser.title)

# ============================
# Cask (A CLI builder)
# ============================

"""
Cask is a simple library for building a CLI application
all in a single file and with no dependencies other than the
Python Standard Library. If you're looking for example look 
for example_app() function.

Copyright (c) 2025, bagasjs
License: MIT (see the details at the very bottom)
"""

from typing import Callable, List, Dict, Any, Tuple
from enum import StrEnum

__author__ = 'bagasjs'
__version__ = '0.0.1'
__license__ = 'MIT'

class ValueType(StrEnum):
    Int    = "Int"
    Float  = "Float"
    String = "String"
    Bool   = "Bool"

class Opt(object):
    def __init__(self, name: str, kind: ValueType, description: str = "", default_value: Any = None, short: str = ""):
        self.name = name
        self.short = short
        self.description = description
        self.kind = kind
        self.default_value = default_value

class Arg(object):
    def __init__(self, name: str, kind: ValueType, default_value: Any):
        self.name = name
        self.kind = kind
        self.default_value = default_value

CommandCallback = Callable[["Command", List[Any], Dict[str, Any]], None]
Error = str | None

def parse_value(value: str, kind: ValueType) -> Tuple[Any, Error]:
    match kind:
        case ValueType.Int:
            try:
                int_value = int(value)
                return int_value, None
            except ValueError as err:
                return 0, str(err)
        case ValueType.Float:
            try:
                float_value = float(value)
                return float_value, None
            except ValueError as err:
                return 0, str(err)
        case ValueType.String:
            return value, None
        case ValueType.Bool:
            match value:
                case "true":
                    return True, None
                case "false":
                    return False, None
                case _:
                    return value, f"Failed to parse boolean value from {value}"
        case _:
            return 0, "Unsupported types"


class Command(object):
    use: str
    description: str
    args: List[Arg]
    opts: List[Opt]

    opt_maps: Dict[str, int]
    subcommands: Dict[str, Command]
    run: CommandCallback | None

    def __init__(self, use: str, description: str, run: 
                 CommandCallback | None = None,
                 opts: List[Opt] | None = None,
                 args: List[Arg] | None = None):
        self.use = use
        self.description = description
        self.args = args if args is not None else []
        self.opts = opts if opts is not None else []
        self.opt_maps = {}
        self.subcommands = {}
        self.run = run
        if "help" not in self.opt_maps:
            self.opts.append(Opt(name="help", kind=ValueType.Bool, description="Get the `usage` information of a command"))

    def add_subcommand(self, command: Command) -> Command:
        self.subcommands[command.use] = command
        return self

    def parse_args(self, args: List[str]) -> Tuple[List[Any], Dict[str, Any], Error]:
        args_length = len(args)
        opts: Dict[str, Any] = {}
        parsed_args: List[Any] = []
        cmd_args_length = len(self.args)

        for i, opt in enumerate(self.opts):
            opts[opt.name] = opt.default_value
            if len(opt.short) != 0:
                self.opt_maps[opt.short] = i
            self.opt_maps[opt.name] = i

        i = 0
        while i < args_length:
            arg = args[i]
            if arg.startswith("-"):
                opt_name  = arg.lstrip("-")
                opt_value = "true" # Default for boolean flags
                if "=" in opt_name:
                    parts = opt_name.split("=", 2)
                    opt_name = parts[0]
                    opt_value = parts[1]
                else:
                    if i + 1 < len(args):
                        opt_value = args[i+1]
                        i += 1


                if opt_name in self.opt_maps:
                    opt_index = self.opt_maps[opt_name]
                    assert type(opt_index) == int
                    parsed_value, err = parse_value(opt_value, self.opts[opt_index].kind)
                    if err is not None:
                        return ([], {}, err)
                    opts[opt_name] = parsed_value
                else:
                    return ([], {}, f"Unknown option: {opt_name}")
            else:
                if len(parsed_args) + 1 <= cmd_args_length:
                    arg_info = self.args[len(parsed_args)]
                    parsed_arg, err = parse_value(arg, arg_info.kind)
                    if err is not None:
                        return ([], {}, err)
                    parsed_args.append(parsed_arg)
            i += 1

        if len(parsed_args) < cmd_args_length:
            return [], {}, "not enough arguments provided"
        return parsed_args, opts, None

    def usage(self):
        print(f"Usage: {self.use} [SUBCOMMANDS] [OPTIONS]", end="")
        for arg in self.args:
            if arg.default_value is not None:
                print(f"  [{arg.name} ({arg.kind}, default {arg.default_value})]", end="")
            else:
                print(f"  <{arg.name} ({arg.kind})>", end="")
                pass
        print(f"\n{self.description}")

        if len(self.opts) > 0:
            print("\nOptions:")
            for opt in self.opts:
                if opt.short != "":
                    print(f"  -{opt.short}, --{opt.name}", end="")
                else:
                    print(f"  --{opt.name}", end="")
                print(f" ({opt.kind})", end="")
                if opt.default_value != None:
                    print(f" [default: {opt.default_value}]", end="")
                if len(opt.description) > 0:
                    print(f" - {opt.description}", end="")
                print()

        if len(self.subcommands) > 0:
            print("\nSubcommands:")
            for name, subcommand in self.subcommands.items():
                print(f" {name}: {subcommand.description}")
        print()

    def execute(self, args: List[str]):
        for i, opt in enumerate(self.opts):
            self.opt_maps[opt.name] = i

        if len(args) == 0 and self.run is None:
            return self.usage()

        if len(args) > 0:
            subcommand_name = args[0]
            if subcommand_name in self.subcommands:
                subcommand = self.subcommands[args[0]]
                return subcommand.execute(args[1:])

        parsed_args, opts, err = self.parse_args(args)
        if err is not None:
            print("ERROR:", err, end="\n\n")
            return self.usage()

        if "help" in opts and opts["help"]:
            return self.usage()

        if self.run:
            self.run(self, parsed_args, opts)
        else:
            self.usage()

def execute(command: Command):
    import sys
    return command.execute(sys.argv[1:])


###########################################################
#
# Downloader APIs
#
###########################################################

# ============================
# Interfaces
# ============================

class Manga(object):
    title: str
    index_url: str
    adapter: str
    pages_url: list[str]

    def __init__(self, title: str, adapter: str, index_url: str, pages_url: list[str]):
        self.title = title
        self.adapter = adapter
        self.index_url = index_url
        self.pages_url = pages_url

class WebClient(ABC):
    """
    Web Client basically the one that will do HTTP requests
    """

    @abstractmethod
    def get(self, url: str) -> tuple[str, bool]:
        pass

class Adapter(ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def search_by_title(self, client: WebClient, title: str) -> tuple[list[Manga], bool]:
        """
        This should return the list of URLs
        i.e.
        [
            "https://mangadex.org/title/0fc02112-a542-4067-a885-ae9dd0503b08/hello-mr-rabbit",
            "https://mangadex.org/title/60bed227-213f-470a-a95e-dd9042b4d5b6/hello-tax-audit",
        ]
        """

    @abstractmethod
    def load_chapters(self, client: WebClient, url: str) -> tuple[list[str], bool]:
        pass

    @abstractmethod
    def download(self, client: WebClient, url: str, librarydir: str) -> bool:
        pass


# ============================
# Spesific Implementation
# ============================

class SimpleWebClient(WebClient):
    def __init__(self):
        self.session = requests.Session()

    def get(self, url: str) -> tuple[bytes, bool]:
        res = self.session.get(url, timeout=30)
        if not res.ok:
            return res.reason.encode(), False
        return res.content, True

###########################################################
#
# Web Based Manga Viewer Implementation
#
###########################################################

class OwnedManga(object):
    title: str
    file_list: list[str]

class HollowViewer(object):
    def __init__(self, owned_mangas: list[OwnedManga]):
        pass

###########################################################
#
# The Main Function
#
###########################################################
HOLLOW_ADAPTERS_DIR = "hollow_adapters"
HOLLOW_LIBRARY_DIR = "hollow_library"

ADAPTERS = {}

def run_list_adapters_command(command: Command, args: list[str], opts: dict[str, Any]):
    print("List of available adapters:")
    for i, key in enumerate(ADAPTERS):
        print("%2d. %s" % (i, key))

def run_load_chapters_command(command: Command, args: list[str], opts: dict[str, Any]):
    query = args[0]
    client = SimpleWebClient()

    query = query.split(":", 1)
    adapter = ADAPTERS.get(query[0])
    if not adapter:
        print(f"Unknown adapter {query[0]}")
        return

    url = query[1]
    print(f"Loading chapters with '{adapter.get_name()}' adapter")
    chapters, ok = adapter.load_chapters(client, url)
    if not ok:
        print(f"Failed to load chapter from \"{url}\" with adapter {adapter.get_name()}")
        return

    for i, chapter in enumerate(chapters):
        print(f"{i}. {adapter.get_name()}:{chapter}")

def run_download_command(command: Command, args: list[str], opts: dict[str, Any]):
    query = args[0]
    client = SimpleWebClient()
    query = query.split(":", 1)
    adapter = ADAPTERS.get(query[0])
    if not adapter:
        print(f"Unknown adapter {query[0]}")
        return

    url = query[1]
    if not adapter.download(client, url, librarydir=HOLLOW_LIBRARY_DIR):
        print(f"Failed to download \"{url}\" with adapter {adapter.get_name()}")

def run_search_command(command: Command, args: list[str], opts: dict[str, Any]):
    query = args[0]
    print("Searching for:", query)
    client = SimpleWebClient()

    results = []
    for adapter_name, adapter_instance in ADAPTERS.items():
        print(f"Query using: {adapter_name} ({adapter_instance})")

        mangas, ok = adapter_instance.search_by_title(client, title=query)
        if not ok:
            print(f"Failed to search \"{query}\" with adapter {adapter_name}")
        else:
            results.extend(mangas)
    print("Found:")
    for i, result in enumerate(results):
        print(f"{i}. {result.title} -> {result.adapter}:{result.index_url}")

def run_serve_command(command: Command, args: list[str], opts: dict[str, Any]):
    pass

def main():
    if os.path.exists(HOLLOW_ADAPTERS_DIR):
        for fname in os.listdir(HOLLOW_ADAPTERS_DIR):
            if fname.endswith(".py") and not fname.startswith("_"):
                path = os.path.join(HOLLOW_ADAPTERS_DIR, fname)
                with open(path, "r") as file:
                    local_ns = {}
                    exec(file.read(), globals(), local_ns)
                    for name, item in local_ns.items():
                        if inspect.isclass(item) and issubclass(item, Adapter) and item is not Adapter:
                            instance = item()
                            ADAPTERS[instance.get_name()] = instance

    cli = Command(use="hollow", description="The Hollow Manga Reader")
    cli.add_subcommand(Command(
        use="search",
        description="Search manga by title",
        args=[
            Arg(name="query", kind=ValueType.String, default_value=None)
        ],
        run=run_search_command))
    cli.add_subcommand(Command(
        use="load-chapters",
        description="Load all the available chapter for a manga",
        args=[
            Arg(name="query", kind=ValueType.String, default_value=None)
        ],
        run=run_load_chapters_command))
    cli.add_subcommand(Command(
        use="list-adapters",
        description="Show list of available adapters",
        run=run_list_adapters_command))
    cli.add_subcommand(Command(
        use="download",
        description="Download manga by `index_url` which is produced by `search` command.",
        opts=[
            Opt(name="dir", 
                kind=ValueType.String, 
                description="Output directory where all manga pages will be downloaded",
                default_value=HOLLOW_LIBRARY_DIR)
        ],
        args=[
            Arg(name="url", kind=ValueType.String, default_value=None)
        ],
        run=run_download_command))
    cli.add_subcommand(Command(
        use="serve",
        description="Serve HTTP server in a library directory",
        args=[
            Arg(name="query", kind=ValueType.String, default_value=None)
        ]))
    cli.add_subcommand(Command(
        use="pack",
        description="Pack all the library into .tar and place into your current working directory",
        opts=[
            Opt(name="compression",
                kind=ValueType.String,
                description="The compression type for the filetype can be gz|bz2|xz or just set it empty")
        ]))
    cli.add_subcommand(Command(
        use="unpack",
        description="Unpack all mangas in the .tar and register everything in the library directory"))
    cli.add_subcommand(Command(
        use="register",
        description="Register a valid manga (directory)"))

    execute(cli)

if __name__ == "__main__":
    main()

