def plot(code_string):
    # Execute the code and get the buffer
    buf = eval(f"({code_string})")

    # Return the image from the buffer
    return buf.getvalue()



code_string = """
import matplotlib.pyplot as plt\n\n# Sample data for sales of a product\nmonths = ['January', 'February', 'March', 'April', 'May']\nsales = [150, 200, 250, 300, 350]\n\nplt.plot(months, sales, marker='o')\nplt.title('Sales of a Product Over Time')\nplt.xlabel('Months')\nplt.ylabel('Sales')\nplt.grid(True)\nplt.show()
"""


image = plot(code_string)
print(image)