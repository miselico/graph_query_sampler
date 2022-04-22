from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import Any, ClassVar, Iterable, Mapping, Optional, Union

DESCRIPTOR: _descriptor.FileDescriptor


class EntityOrLiteral(_message.Message):
    __slots__ = ["entity", "literal"]
    ENTITY_FIELD_NUMBER: ClassVar[int]
    LITERAL_FIELD_NUMBER: ClassVar[int]
    entity: int
    literal: str
    def __init__(self, entity: Optional[int] = ..., literal: Optional[str] = ...) -> None: ...


class Qualifier(_message.Message):
    __slots__ = ["corresponding_triple", "qualifier_relation", "qualifier_value"]
    CORRESPONDING_TRIPLE_FIELD_NUMBER: ClassVar[int]
    QUALIFIER_RELATION_FIELD_NUMBER: ClassVar[int]
    QUALIFIER_VALUE_FIELD_NUMBER: ClassVar[int]
    corresponding_triple: int
    qualifier_relation: int
    qualifier_value: EntityOrLiteral
    def __init__(self, qualifier_relation: Optional[int] = ..., qualifier_value: Optional[Union[EntityOrLiteral, Mapping[Any, Any]]] = ..., corresponding_triple: Optional[int] = ...) -> None: ...


class Query(_message.Message):
    __slots__ = ["diameter", "easy_targets", "hard_targets", "qualifiers", "triples"]
    DIAMETER_FIELD_NUMBER: ClassVar[int]
    EASY_TARGETS_FIELD_NUMBER: ClassVar[int]
    HARD_TARGETS_FIELD_NUMBER: ClassVar[int]
    QUALIFIERS_FIELD_NUMBER: ClassVar[int]
    TRIPLES_FIELD_NUMBER: ClassVar[int]
    diameter: int
    easy_targets: _containers.RepeatedCompositeFieldContainer[EntityOrLiteral]
    hard_targets: _containers.RepeatedCompositeFieldContainer[EntityOrLiteral]
    qualifiers: _containers.RepeatedCompositeFieldContainer[Qualifier]
    triples: _containers.RepeatedCompositeFieldContainer[Triple]
    def __init__(self, triples: Optional[Iterable[Union[Triple, Mapping[Any, Any]]]] = ..., qualifiers: Optional[Iterable[Union[Qualifier, Mapping[Any, Any]]]] = ..., easy_targets: Optional[Iterable[Union[EntityOrLiteral, Mapping[Any, Any]]]] = ..., hard_targets: Optional[Iterable[Union[EntityOrLiteral, Mapping[Any, Any]]]] = ..., diameter: Optional[int] = ...) -> None: ...


class QueryData(_message.Message):
    __slots__ = ["queries"]
    QUERIES_FIELD_NUMBER: ClassVar[int]
    queries: _containers.RepeatedCompositeFieldContainer[Query]
    def __init__(self, queries: Optional[Iterable[Union[Query, Mapping[Any, Any]]]] = ...) -> None: ...


class Triple(_message.Message):
    __slots__ = ["object", "predicate", "subject"]
    OBJECT_FIELD_NUMBER: ClassVar[int]
    PREDICATE_FIELD_NUMBER: ClassVar[int]
    SUBJECT_FIELD_NUMBER: ClassVar[int]
    object: EntityOrLiteral
    predicate: int
    subject: int
    def __init__(self, subject: Optional[int] = ..., predicate: Optional[int] = ..., object: Optional[Union[EntityOrLiteral, Mapping[Any, Any]]] = ...) -> None: ...
