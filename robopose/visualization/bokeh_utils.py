import bokeh
from bokeh.plotting import figure as bokeh_figure
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import DataTable, TableColumn
from bokeh.models.widgets import NumberFormatter
import bokeh.io
import numpy as np
from PIL import Image
from pathlib import Path
from bokeh.io.export import get_screenshot_as_png

def save_image_figure(f, jpg_path):
    f.toolbar.logo = None
    f.toolbar_location = None
    f.title = None
    f.sizing_mode = 'fixed'
    im = get_screenshot_as_png(f)
    w, h = im.size
    im = im.convert('RGB').crop((1, 1, w, h)).resize((w, h))
    im.save(jpg_path)
    return im

def to_rgba(im):
    im = Image.fromarray(im)
    im = np.asarray(im.convert('RGBA'))
    im = np.flipud(im)
    return im


def plot_image(im, axes=False, tools='', im_size=None, figure=None):
    if np.asarray(im).ndim == 2:
        gray = True
    else:
        im = to_rgba(im)
        gray = False

    if im_size is None:
        h, w = im.shape[:2]
    else:
        h, w = im_size
    source = bokeh.models.sources.ColumnDataSource(dict(rgba=[im]))
    f = image_figure('rgba', source, im_size=(h, w), axes=axes, tools=tools, gray=gray, figure=figure)
    return f, source


def make_image_figure(im_size=(240, 320), axes=False):
    w, h = max(im_size), min(im_size)
    f = bokeh_figure(x_range=(0, w-1), y_range=(0, h-1),
                     plot_width=w, plot_height=h, title='',
                     tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])
    # f.x_range.range_padding = f.y_range.range_padding = 0
    f.toolbar.logo = None
    f.toolbar_location = None
    f.axis.visible = False
    f.grid.visible = False
    f.min_border = 0
    f.outline_line_width = 0
    f.outline_line_color = None
    f.background_fill_color = None
    f.border_fill_color = None
    return f


def image_figure(key, source, im_size=(240, 320), axes=False, tools='',
                 gray=False, figure=None):
    h, w = im_size
    if figure is None:
        # f = bokeh_figure(x_range=(0, w-1), y_range=(0, h-1),
        #                  plot_width=w, plot_height=h, tools=tools,
        #                  tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])
        f = make_image_figure(im_size=im_size, axes=axes)
    else:
        f = figure

    if gray:
        f.image(key, x=0, y=0, dw=w, dh=h, source=source)
    else:
        f.image_rgba(key, x=0, y=0, dw=w, dh=h, source=source)
    return f


def convert_df(df):
    columns = []
    for column in df.columns:
        if df.dtypes[column].kind == 'f':
            formatter =  NumberFormatter(format='0.000')
        else:
            formatter = None
        table_col = TableColumn(field=column, title=column, formatter=formatter)
        columns.append(table_col)
    data_table = DataTable(columns=columns, source=ColumnDataSource(df), height=200)
    return data_table
