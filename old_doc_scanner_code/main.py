from flask import Flask, request, render_template
import settings
import utils
import numpy as np
import cv2


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
        if four_points is None:
            message = "Unable to locate the corodinate of document"
            points = [
                {'x': 10, 'y': 10},
                {'x': 120, 'y': 10},
                {'x': 120, 'y': 120},
                {'x': 10, 'y': 120}
            ]
            return render_template('scanner.html', 
                                    points=points,
                                    fileupload = True,
                                    message = message
                                    )
        else:
            points = utils.array_to_json_format(four_points)
            message = "Located the Corridinates of Document using OPenCv"
            return render_template('scanner.html',
                                    points=points,
                                    fileupload = True,
                                    message = message)
            
        return render_template('scanner.html')
    
    return render_template('scanner.html')

@app.route('/transform', methods=['POST'])
def transform():
    try:
        points = request.json['data']
        array = np.array(points)
        magic_color = docscan.calibrate_to_original_size(array)
        # utils.save_image(magic_color, 'magic_color.jpg')

        filename = 'magic_color.jpg'
        magic_image_path = settings.join_path(settings.MEDIA_DIR, filename)
        cv2.imwrite(magic_image_path, magic_color)
        
        return 'sucess'
    except:
        return 'fail'

@app.route("/prediction")
def prediction():
    return 'successfully wrap the image'

@app.route('/about')
def about():
    return render_template('about.html')










if __name__ == "__main__":
    app.run(debug=True)