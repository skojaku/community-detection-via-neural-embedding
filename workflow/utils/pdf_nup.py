"""Arrange multiple single-page PDFs into a grid on one page (replaces pdfjam --nup)."""
import argparse
from pypdf import PdfReader, PdfWriter, PageObject, Transformation


def nup(input_files, output_file, cols, rows):
    pages = []
    for f in input_files:
        reader = PdfReader(f)
        pages.append(reader.pages[0])

    widths = [float(p.mediabox.width) for p in pages]
    heights = [float(p.mediabox.height) for p in pages]

    cell_w = max(widths)
    cell_h = max(heights)
    total_w = cell_w * cols
    total_h = cell_h * rows

    new_page = PageObject.create_blank_page(width=total_w, height=total_h)

    for idx, page in enumerate(pages):
        col = idx % cols
        row = idx // cols
        if row >= rows:
            break
        tx = col * cell_w
        # PDF origin is bottom-left; row 0 should be at the top
        ty = total_h - (row + 1) * cell_h
        new_page.merge_transformed_page(
            page, Transformation().translate(tx=tx, ty=ty)
        )

    writer = PdfWriter()
    writer.add_page(new_page)
    with open(output_file, "wb") as fh:
        writer.write(fh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="Input PDF files")
    parser.add_argument("--nup", required=True, help="Grid as CxR, e.g. 1x3")
    parser.add_argument("--outfile", required=True, help="Output PDF path")
    args = parser.parse_args()
    cols, rows = (int(x) for x in args.nup.split("x"))
    nup(args.inputs, args.outfile, cols, rows)
