from openpyxl import load_workbook


def write_to_excel(filename, sheetname, cell, data):
    # load the workbook
    workbook = load_workbook(filename)

    # select the sheet
    sheet = workbook[sheetname]

    # write data to the cell
    sheet[cell] = data

    # save the workbook
    workbook.save(filename)







