import enum


class ContactType(enum.IntEnum):
    UNSPECIFIED = 0
    SELECTED = 1


class PocketContactType(enum.IntEnum):
    UNSPECIFIED = 0
    UNSELECTED = 1
    POCKET = 2
    BINDER = 3