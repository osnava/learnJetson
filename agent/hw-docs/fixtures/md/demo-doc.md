# demo-doc (PDF)

Synthetic stand-in for a converted corpus document (same shape as
pymupdf4llm output: `<!-- p.N -->` anchors, `## N.M` sections, `# **Chapter N. …**`
chapters, pipe tables with `<br>` cell wraps, plain-line table captions,
and a running-header page that carries no section content).

<!-- p.1 -->

Chapter One

# **Chapter 1. Widgets**

## 1.1 Widget Basics

Body prose about widgets. The widget supply rail is 3.3 V nominal and the
ready line idles high.

<!-- p.2 -->

Running header text only.

## 1.2 Widget Modes

Table 1-1. Widget Mode Pins – J9

|**Pin**||**Mode**|
|---|---|---|
|1|WIDGET_RDY|Ready signal, 3.3V|
|2|WIDGET_ERR: Multi word error text<br>continues on this wrapped cell|Input, 1.8V|

<!-- p.3 -->

Running header text only.

## 1.3 Widget Limits

Absolute maximum is 125 degrees C at full load.

<!-- p.4 -->

Running header text only.

<!-- p.5 -->

Trailing filler page with closing remarks. More filler text here so the
last page carries real content past its running header and nothing ends
abruptly at an anchor boundary.
