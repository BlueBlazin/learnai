import json


def schema_to_regex(schema: dict) -> str:
    """A regular expression that matches exactly the compact JSON documents of `schema`.

    schema: a JSON Schema of the statement's subset (null, boolean, integer, number,
        string with minLength/maxLength, enum, array with items/minItems/maxItems,
        object with properties/required), nested to any depth.
    Returns a pattern for re.fullmatch (no flags), written in the regex subset of
    `regex-to-dfa`: no anchors, lookaround, backreferences, or lazy quantifiers.
    Raises ValueError for a schema without `enum` whose `type` is not one of the
    seven supported type names.
    """
    raise NotImplementedError
