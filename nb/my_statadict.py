#!/usr/bin/env python3

from typing import List
from collections import deque
import re


class StataDict:
    """
    Class representing Stata dictionary file.

    Consists of several attributes describing columns in a fixed width field csv file.

    'names', 'colspecs' and 'widths' attributes can be used with 'pandas.read_fwf()'.
    """
    def __init__(
            self,
            column_numbers: List[int],
            types: List[str],
            names: List[str],
            formats: List[str],
            comments: List[str],
    ):
        self._column_numbers = column_numbers
        self._types = types
        self._names = names
        self._formats = formats
        self._comments = comments
        self._widths = None
        self._colspecs = None

    @property
    def column_numbers(self) -> List[int]:
        """
        :return: Number for each column where that column data should start in fwf file (1-indexed)
        :rtype: List[int]
        """
        return self._column_numbers

    @property
    def types(self) -> List[str]:
        """
        :return: Data type for each column
        :rtype: List[str]
        """
        return self._types

    @property
    def names(self) -> List[str]:
        """
        This attribute can be used with pandas.read_fwf() as names argument.

        :return: Name for each column
        :rtype: List[str]
        """
        return self._names

    @property
    def formats(self) -> List[str]:
        """
        :return: Parsing format for each column
        :rtype: List[str]
        """
        return self._formats

    @property
    def comments(self) -> List[str]:
        """
        :return: Optional comment for each column
        :rtype: List[str]
        """
        return self._comments

    @property
    def widths(self) -> List[int]:
        """
        This attribute can be used with pandas.read_fwf() as widths argument.

        :return: Width for each column
        :rtype: List[int]
        """
        if not self._widths:
            widths = deque()
            for i in range(len(self._formats)):
                widths.append(int(re.findall(r"\d+", self._formats[i])[0]))
            self._widths = list(widths)
        return self._widths

    @property
    def colspecs(self) -> List[tuple]:
        """
        This attribute can be used with pandas.read_fwf() as colspecs argument.

        :return: Tuple with start and end of data for each column
        :rtype: List[tuple]
        """
        if not self._colspecs:
            colspecs = deque()
            for i in range(len(self._column_numbers)):
                colspecs.append((
                    self._column_numbers[i] - 1,
                    self._column_numbers[i] + self.widths[i] - 1
                ))
            self._colspecs = list(colspecs)
        return self._colspecs


class StataDictParser:
    _COLUMN_PATTERN = r'^\s+_column'
    _LINE_PATTERN = r'^\s+_column\((\d+)\)\s+(\S+)\s+(\S+)\s+(\S+)\s*(".*")?'
    print(_LINE_PATTERN)

    def parse(self, file) -> StataDict:
        column_numbers = deque()
        types = deque()
        names = deque()
        formats = deque()
        comments = deque()
        with open(file, "r") as dct_file:
            for line in dct_file:
                if re.search(self._COLUMN_PATTERN, line):
                    line_values = re.findall(self._LINE_PATTERN, line)
                    column_numbers.append(int(line_values[0][0]))
                    types.append(line_values[0][1])
                    names.append(line_values[0][2])
                    formats.append(line_values[0][3])
                    if len(line_values[0]) == 5:
                        comments.append(line_values[0][4].lstrip('"').rstrip('"'))
                    else:
                        comments.append(None)
                else:
                    print(line)
        return StataDict(
            column_numbers=list(column_numbers),
            types=list(types),
            names=list(names),
            formats=list(formats),
            comments=list(comments)
        )


def parse_stata_dict(file: str) -> StataDict:
    """
    Parses Stata dictionary file and returns object containing column data as attributes.

    'names', 'colspecs' and 'widths' attributes can be used with 'pandas.read_fwf()'.

    :param file: Stata dictionary file (usually .dct extension)
    :type: str
    :return: Object with column data as attributes
    :rtype: statadict.base.StataDict
    """
    stata_dict_parser = StataDictParser()
    return stata_dict_parser.parse(file)
