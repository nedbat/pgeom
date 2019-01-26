import contextlib
from io import BytesIO
import math

import attr
import cairo
import IPython.display
import numpy as np
import transforms3d.affines
import transforms3d.axangles as t3ax


@attr.s(frozen=True)
class Point3:
    """A point in three dimensions."""
    x = attr.ib()
    y = attr.ib()
    z = attr.ib()

    def xyz(self):
        return attr.astuple(self)

    def transform(self, aff):
        xyz_ = self.xyz() + (1,)
        txyz_ = np.dot(aff, xyz_)
        return Point3(*txyz_[:3])

    def is_close(self, p2):
        return (
            math.isclose(self.x, p2.x) and
            math.isclose(self.y, p2.y) and
            math.isclose(self.z, p2.z)
            )

def hypot3(a, b, c):
    return math.hypot(math.hypot(a, b), c)

@attr.s(frozen=True)
class Vector3:
    """A vector in three dimensions."""
    dx = attr.ib()
    dy = attr.ib()
    dz = attr.ib()

    @classmethod
    def from_points(cls, p1, p2):
        """The vector from `p1` to `p2`."""
        x1, y1, z1 = attr.astuple(p1)
        x2, y2, z2 = attr.astuple(p2)
        return cls(x2 - x1, y2 - y1, z2 - z1)

    def dxdydz(self):
        return attr.astuple(self)

    def magnitude(self):
        return hypot3(self.dx, self.dy, self.dz)

    def unit(self):
        """Same direction as self, but unit length."""
        hyp = self.magnitude()
        return Vector3(self.dx / hyp, self.dy / hyp, self.dz / hyp)

    def cross(self, other):
        """Cross product of self with other."""
        sx, sy, sz = self.dxdydz()
        ox, oy, oz = other.dxdydz()
        return Vector3(
            sy * oz - sz * oy,
            sx * oz - sz * ox,
            sx * oy - sy * ox,
            )
    
    def dot(self, other):
        """Dot product of self with other."""
        sx, sy, sz = self.dxdydz()
        ox, oy, oz = other.dxdydz()
        return sx * ox + sy * oy + sz * oz


@attr.s(frozen=True)
class Line3:
    """A line in three dimensions."""
    p1 = attr.ib(type=Point3)
    p2 = attr.ib(type=Point3)

    def p1p2(self):
        return self.p1, self.p2


@attr.s(frozen=True)
class Plane:
    """A plane in three dimensions."""
    # Represented in point-normal form: a point in the plane, and the unit normal.
    pt = attr.ib(type=Point3)
    unormal = attr.ib(type=Vector3)

    @classmethod
    def from_points(cls, p1, p2, p3):
        v2 = Vector3.from_points(p1, p2)
        v3 = Vector3.from_points(p1, p3)
        normal = v2.cross(v3)
        return cls(pt=p1, unormal=normal.unit())

    def abcd(self):
        a, b, c = self.unormal.dxdydz()
        x0, y0, z0 = self.pt.xyz()
        d = a * x0 + b * y0 + c * z0
        return a, b, c, d

    def distance_to_point(self, pt):
        return self.unormal.dot(Vector3.from_points(self.pt, pt))

    def is_parallel(self, other):
        dot = self.unormal.dot(other.unormal)
        return math.isclose(abs(dot), 1)

    def intersect_plane(self, other):
        """Returns two Point3's on the intersection."""
        a = self.abcd()
        b = other.abcd()

        # From: https://stackoverflow.com/questions/48126838/plane-plane-intersection-in-python
        a_vec, b_vec = np.array(a[:3]), np.array(b[:3])
        aXb_vec = np.cross(a_vec, b_vec)
        A = np.array([a_vec, b_vec, aXb_vec])
        d = np.array([-a[3], -b[3], 0.]).reshape(3,1)

        # could add np.linalg.det(A) == 0 test to prevent linalg.solve throwing error
        p_inter = np.linalg.solve(A, d).T
        return Line3(Point3(*p_inter[0]), Point3(*(p_inter + aXb_vec)[0]))



@attr.s(frozen=True, cmp=False)
class Segment3:
    """Line segment defined by two Point3's"""
    p1 = attr.ib(type=Point3)
    p2 = attr.ib(type=Point3)

    def __eq__(self, other):
        return sorted(attr.astuple(self)) == sorted(attr.astuple(other))

    def __hash__(self):
        return hash(tuple(sorted(attr.astuple(self))))


@attr.s(frozen=True)
class Bounds3:
    lx = attr.ib()
    ux = attr.ib()
    ly = attr.ib()
    uy = attr.ib()
    lz = attr.ib()
    uz = attr.ib()

    @classmethod
    def from_points(cls, pts):
        return cls(
            min(pt.x for pt in pts),
            max(pt.x for pt in pts),
            min(pt.y for pt in pts),
            max(pt.y for pt in pts),
            min(pt.z for pt in pts),
            max(pt.z for pt in pts),
        )

    def __or__(self, other):
        return self.__class__(
            min(self.lx, other.lx),
            max(self.ux, other.ux),
            min(self.ly, other.ly),
            max(self.uy, other.uy),
            min(self.lz, other.lz),
            max(self.uz, other.uz),
        )

    def center(self):
        return Point3(
            (self.lx + self.ux) / 2,
            (self.ly + self.uy) / 2,
            (self.lz + self.uz) / 2,
        )


def d2r(d):
    return d / 180.0 * math.pi

class Drawing:
    def __init__(self, width, height, bounds=None, debug_origin=False):
        self.svgio = BytesIO()
        self.surface = cairo.SVGSurface(self.svgio, width, height)
        self.ctx = cairo.Context(self.surface)
        self.ctx.set_line_cap(cairo.LineCap.ROUND)
        self.ctx.set_line_join(cairo.LineJoin.MITER)

        if bounds:
            lx, ly, ux, uy = bounds
            sx = width / (ux - lx)
            sy = height / (uy - ly)
            sfactor = min(sx, sy)
            self.scale(sfactor, sfactor)
            self.translate(-lx, -ly)
        else:
            sfactor = 1

        if debug_origin:
            self.origin_rays()

    def origin_rays(self):
        r = 1000
        linewidth = self.screen_units(1)
        for theta in range(0, 360, 2):
            with self.style(rgb=(.9, .9, 1), width=linewidth):
                self.move_to(0, 0)
                self.line_to(
                    math.cos(d2r(theta)) * r,
                    math.sin(d2r(theta)) * r,
                )
                self.stroke()

    def grid(self):
        r = 100
        lblue = (.9, .9, 1)
        with self.style(rgb=lblue):
            for xy in range(-r, r):
                if xy == 0:
                    width = 3.5
                elif xy % 5 == 0:
                    width = 2
                else:
                    width = .8
                with self.style(width=self.screen_units(width)):
                    self.move_to(xy, -r)
                    self.line_to(xy, r)
                    self.move_to(-r, xy)
                    self.line_to(r, xy)
                    self.stroke()

    def screen_units(self, d):
        """Compute user-space distance for screen distance `d`."""
        ox, _ = self.ctx.device_to_user(0, 0)
        dx, _ = self.ctx.device_to_user(d, d)
        return dx - ox

    def __getattr__(self, name):
        """Use the drawing like a context, or a surface."""
        try:
            return getattr(self.ctx, name)
        except AttributeError:
            return getattr(self.surface, name)

    @contextlib.contextmanager
    def style(self, rgb=None, rgba=None, width=None, dash=None, dash_offset=0):
        """Set and restore the drawing style."""
        o_source = self.get_source()
        o_width = self.get_line_width()
        o_dash = self.get_dash()
        try:
            if rgb is not None:
                self.set_source_rgb(*rgb)
            if rgba is not None:
                self.set_source_rgba(*rgba)
            if width is not None:
                self.set_line_width(width)
            if dash is not None:
                self.set_dash(dash, dash_offset)
            yield
        finally:
            self.set_source(o_source)
            self.set_line_width(o_width)
            self.set_dash(*o_dash)

    def display(self):
        self.surface.flush()
        self.surface.finish()
        return IPython.display.SVG(data=self.svgio.getvalue())

def draw_cross(drawing, x, y, r):
    drawing.move_to(x-r, y)
    drawing.line_to(x+r, y)
    drawing.move_to(x, y-r)
    drawing.line_to(x, y+r)
    drawing.stroke()

def draw_line(drawing, line):
    p1, p2 = line.p1p2()
    (x1, y1, _), (x2, y2, _) = p1.xyz(), p2.xyz()
    dx = x2 - x1
    dy = y2 - y1
    n = 100
    drawing.move_to(x1 - n * dx, y1 - n * dy)
    drawing.line_to(x2 + n * dx, y2 + n * dy)


def transform_plane_to_xy(plane):
    normal = plane.unormal
    up = Vector3(0, 0, 1)
    angle = math.acos(normal.dot(up))
    axis = up.cross(normal)
    mrotate = transforms3d.axangles.axangle2aff(axis.dxdydz(), angle)
    rotated = plane.pt.transform(mrotate)
    mraise = transforms3d.affines.compose([0, 0, -rotated.z], np.eye(3), np.ones(3))
    m = np.dot(mraise, mrotate)
    return m

def sit_polyhedron(poly):
    t = list(map(lambda v: -v, poly.bounds().center().xyz()))
    mtranslate = transforms3d.affines.compose(t, np.eye(3), np.ones(3))
    poly = poly.transform(mtranslate)
    plane = poly.planes()[0]
    mrotate = transform_plane_to_xy(plane)
    poly = poly.transform(mrotate)
    return poly


class Polyhedron:
    def __init__(self, name, faces):
        self.name = name
        self.faces = faces

    def __repr__(self):
        return f"<Polyhedron {self.name!r} {len(self.faces)} faces>"

    def bounds(self):
        return Bounds3.from_points(self.vertices())

    def planes(self):
        return [Plane.from_points(*f[:3]) for f in self.faces]

    def vertices(self):
        return set(pt for f in self.faces for pt in f)

    def edges(self):
        e = set()
        for f in self.faces:
            for p1, p2 in zip(f, f[1:]):
                e.add(Segment3(p1, p2))
            e.add(Segment3(f[-1], f[0]))
        return e

    def transform(self, aff):
        tfaces = [[p.transform(aff) for p in f] for f in self.faces]
        return Polyhedron(self.name, tfaces)

def wire_frame(poly, **drawing_kwargs):
    drawing_kwargs.setdefault('width', 200)
    drawing_kwargs.setdefault('height', 200)
    b3 = poly.bounds()
    margin = (b3.ux - b3.lx) * 0.1
    bounds = (b3.lx - margin, b3.ly - margin, b3.ux + margin, b3.uy + margin)
    drawing = Drawing(bounds=bounds, **drawing_kwargs)
    drawing.set_line_width(0.02)
    for seg in poly.edges():
        drawing.move_to(seg.p1.x, seg.p1.y)
        drawing.line_to(seg.p2.x, seg.p2.y)
        drawing.stroke()
    IPython.display.display(drawing.display())

def read_netlib(lines):
    # Make a dict of sections by name
    sects = {}
    cursect = None
    for line in lines:
        line = line.rstrip()
        if line.startswith(':'):
            sects[line[1:]] = cursect = []
        else:
            cursect.append(line)

    # Get the solid vertex indices
    face_vertices = []
    vertex_indexes = set()
    solid_lines = iter(sects['solid'])
    num_verts, max_verts = map(int, next(solid_lines).split())
    assert len(sects['solid']) == num_verts + 1
    for line in solid_lines:
        ints = list(map(int, line.split()))
        assert len(ints) == ints[0] + 1
        assert ints[0] <= max_verts
        face_vertices.append(ints[1:])
        vertex_indexes.update(ints[1:])

    # Get the 3d vertices
    vertices = []
    vertex_lines = sects['vertices'][1:]
    vertex_map = {}
    for vi in vertex_indexes:
        vertex = [float(v.partition('[')[0]) for v in vertex_lines[vi].split()]
        vertex_map[vi] = Point3(*vertex)
    faces = [
        list(map(vertex_map.get, face))
        for face in face_vertices
        ]
    name = sects['name'][0]
    return Polyhedron(name, faces)
