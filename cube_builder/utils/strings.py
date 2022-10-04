#
# This file is part of Cube Builder.
# Copyright (C) 2022 INPE.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/gpl-3.0.html>.
#

"""Define a module to deal with string functionalities.

This module extends the builtin Python string module.
"""

import string


class StringFormatter(string.Formatter):
    """Implement a string formatter and supports pipes to format string.

    Examples:
        >> from cube_builder.utils.strings import StringFormatter
        >> formatter = StringFormatter()
        >> template = '{name:upper}, {name:lower}, {name:capitalize}'
        >> name = 'CustomNameTiler'
        >> assert formatter.format(template, name=name) == f'{name.upper()}, {name.lower()}, {name.capitalize()}'
    """

    DEFAULT_STRING_FUNCTIONS = ['upper', 'lower', 'capitalize']
    """List of supported generic string functions."""

    def format_field(self, value, format_spec):
        """Format a string according PEP3101."""
        if isinstance(value, str):
            if format_spec in self.DEFAULT_STRING_FUNCTIONS:
                handler = self.DEFAULT_STRING_FUNCTIONS[self.DEFAULT_STRING_FUNCTIONS.index(format_spec)]
                value = getattr(value, handler)()
                format_spec = format_spec[:-1]
        elif isinstance(value, int) or isinstance(value, float):
            value = str(value)
        return super(StringFormatter, self).format(value, format_spec)
