from flask import Flask, request, render_template
import settings
import utils


app = Flask(__name__)
app.secret_key = 'faruk'

docscan = utils.DocumentScan()

@app.route('/', methods=['GET', 'POST'])
def scandoc():
    # save the image
    if request.method == 'POST':
        file = request.files['image_name']
        upload_image_path = utils.save_upload_image(file)
        print("Image save in = ", upload_image_path)
        # predict the coordinate the document
        four_points, size = docscan.document_scanner(upload_image_path)
        print(four_points, size)
        
        return render_template('scanner.html')
    
    return render_template('scanner.html')

@app.route('/about')
def about():
    return render_template('about.html')










if __name__ == "__main__":
    app.run(debug=True)