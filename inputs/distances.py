import googlemaps
import numpy as np
import openpyxl
from openpyxl import load_workbook


GMAPS = googlemaps.Client(key='AIzaSyDDgmthv161tSfmFVmglxlEuJsxn7WnH9A')


class Location:
    def __init__(self, name, latitude, longitude):
        self.name = name
        self.latitude = latitude
        self.longitude = longitude

    def __repr__(self):
        return f"({self.latitude}, {self.longitude}: {self.name}"


def create_locations(filename: str):
    workbook = load_workbook(filename=filename)
    sheet = workbook.active
    locations = []

    for row in range(2, sheet.max_row + 1):
        name = sheet.cell(row=row, column=1).value
        latitude = sheet.cell(row=row, column=9).value
        longitude = sheet.cell(row=row, column=10).value

        # Create a Location object and append it to the list
        location = Location(name, latitude, longitude)
        locations.append(location)

    return locations


def create_distance_matrix(locations: list[Location], max_elements=10):
    n = len(locations)
    distance_matrix = [[0] * n for _ in range(n)]
    coordinates = [(location.latitude, location.longitude) for location in locations]

    for i in range(0, n, max_elements):
        for j in range(0, n, max_elements):
            origins = coordinates[i:min(i + max_elements, n)]
            destinations = coordinates[j:min(j + max_elements, n)]

            # Call the Distance Matrix API with the current set of origins and destinations
            distance_matrix_response = GMAPS.distance_matrix(origins, destinations, mode="driving")

            for k, row in enumerate(distance_matrix_response["rows"]):
                for l, element in enumerate(row["elements"]):
                    distance_matrix[i + k][j + l] = element["distance"]["value"]

    return distance_matrix


def write_distance_matrix_to_excel(distance_matrix, file_name, sheet_name):
    # Load the existing workbook
    workbook = openpyxl.load_workbook(file_name)

    # Check if the sheet_name exists in the workbook, if not, create it
    if sheet_name not in workbook.sheetnames:
        workbook.create_sheet(sheet_name)

    # Select the sheet
    sheet = workbook[sheet_name]

    # Write the distance matrix to the sheet
    for i, row in enumerate(distance_matrix, start=2):  # Start at row 2
        for j, distance in enumerate(row, start=2):  # Start at column B
            sheet.cell(row=i, column=j, value=distance)

    # Save the workbook
    workbook.save(file_name)


def make_symmetric(matrix: list[list[int]]):
    n = len(matrix)
    for i in range(n):
        for j in range(i+1, n):
            # use the row value to update the column
            matrix[j][i] = matrix[i][j]
    return matrix


def create_symmetric_distance_matrix(input_file: str, output_file: str, output_sheet: str):
    locations = create_locations(input_file)
    matrix = create_distance_matrix(locations)
    matrix = make_symmetric(matrix)
    write_distance_matrix_to_excel(matrix, output_file, output_sheet)


def read_in_distance_matrix(input_file: str, input_sheet: str, topleft: str, bottomright: str):
    # Load the workbook
    workbook = load_workbook(filename=input_file, read_only=True)

    # Select the specified sheet
    sheet = workbook[input_sheet]

    # Select the specified range of cells
    cell_range = sheet[topleft:bottomright]

    # Read the values of the cells into a nested list (distance matrix)
    distance_matrix = [[cell.value for cell in row] for row in cell_range]
    distance_matrix = np.array(distance_matrix)

    return distance_matrix


def normalise_geo_coordinates(input_file: str, new_origin: tuple):
    locations = create_locations(input_file)

    origin_latitude, origin_longitude = new_origin
    for location in locations:
        location.latitude = round(location.latitude - origin_latitude, 6)
        location.longitude = round(location.longitude - origin_longitude, 6)

    return locations


#create_symmetric_distance_matrix("distances.xlsx", "distances.xlsx", "Sheet2")
#matrix = read_in_distance_matrix("distances.xlsx", "Distance matrix (districts)", "B2", "AX50")





