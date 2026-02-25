"""Arrange multiple single-page PDFs into a grid on one page (replaces pdfjam --nup)."""

import argparse

from pypdf import PageObject, PdfReader, PdfWriter, Transformation


def arrange_pdfs_in_grid(input_files, output_file, cols, rows):
    """Read the first page from each input PDF and tile them into a cols x rows grid."""
    pages = [PdfReader(filepath).pages[0] for filepath in input_files]

    cell_width = max(float(page.mediabox.width) for page in pages)
    cell_height = max(float(page.mediabox.height) for page in pages)
    grid_width = cell_width * cols
    grid_height = cell_height * rows

    grid_page = PageObject.create_blank_page(width=grid_width, height=grid_height)

    max_cells = cols * rows
    for idx, page in enumerate(pages[:max_cells]):
        col = idx % cols
        row = idx // cols
        # PDF origin is bottom-left; row 0 should appear at the top
        tx = col * cell_width
        ty = grid_height - (row + 1) * cell_height
        grid_page.merge_transformed_page(
            page, Transformation().translate(tx=tx, ty=ty)
        )

    writer = PdfWriter()
    writer.add_page(grid_page)
    with open(output_file, "wb") as out_fh:
        writer.write(out_fh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="Input PDF files")
    parser.add_argument("--nup", required=True, help="Grid as CxR, e.g. 1x3")
    parser.add_argument("--outfile", required=True, help="Output PDF path")
    args = parser.parse_args()
    cols, rows = (int(x) for x in args.nup.split("x"))
    arrange_pdfs_in_grid(args.inputs, args.outfile, cols, rows)
