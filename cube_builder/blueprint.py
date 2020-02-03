"""
Brazil Data Cube Scripts Blueprint strategy
"""

from flask import Blueprint
from flask_restplus import Api
from .controller import api as cube_ns


bp = Blueprint('cubes', __name__, url_prefix='/api')

api = Api(bp, doc=False)

api.add_namespace(cube_ns)