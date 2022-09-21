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

"""Represent the data cube parameters to be attached in execution step."""

import sqlalchemy as sa
from bdc_catalog.models import Collection
from bdc_catalog.models.base_sql import BaseModel
from sqlalchemy.orm import relationship

from cube_builder.config import Config


class CubeParameters(BaseModel):
    """Define a table to store the data cube parameters to be passed during the execution."""

    __tablename__ = 'cube_parameters'

    id = sa.Column(sa.Integer, primary_key=True)
    collection_id = sa.Column(sa.ForeignKey(Collection.id, onupdate='CASCADE', ondelete='CASCADE'),
                              nullable=False)
    metadata_ = sa.Column(sa.JSON, default='{}', nullable=False)

    cube = relationship(Collection, lazy='select')

    __table_args__ = (
        sa.Index(None, collection_id),
        dict(schema=Config.ACTIVITIES_SCHEMA),
    )

    def _require_property(self, prop: str):
        if self.metadata_.get(prop) is None:
            raise RuntimeError(f'Missing property "{prop}" in data cube parameters {self.id} '
                               f'for data cube "{self.cube.id}"')

    def _check_reuse_cube(self):
        self._reuse_cube = None

        if self.metadata_.get('reuse_data_cube'):
            self._reuse_cube = Collection.query().get(self.metadata_['reuse_data_cube'])
            # TODO: Check bands and band resolution should be equal

    def validate(self):
        """Validate minimal properties for metadata field."""
        if self.cube.composite_function.alias != 'IDT':
            self._require_property('mask')
            self._require_property('quality_band')
        self._check_reuse_cube()

    @property
    def reuse_cube(self):
        """Retrieve the data cube reference to reuse."""
        return self._reuse_cube
