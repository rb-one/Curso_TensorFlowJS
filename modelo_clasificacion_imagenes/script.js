let net;

const imgEl = document.getElementById('img');
const descEl = document.getElementById('descripcion_imagen');
const webcamElement = document.getElementById('webcam');
const classifier = knnClassifier.create();


async function app() {

    // cargamos todos los parametros de mobilenet
    // tiene definidos con su entrenamiento
    net = await mobilenet.load();

    // Método classify 
    var result = await net.classify(imgEl);
    console.log(result);
    displayImagePrediction();

    webcam = await tf.data.webcam(webcamElement);

    while(true){
        const img = await webcam.capture();

        const activation = net.infer(img, "conv_preds");

        var result2;
        
        // creamos red en tiempo real
        try {
            result2 = await classifier.predictClass(activation);
            const classes = ["Untrained", "Gato", "Dino", "Alex", "OK", "Rock"]
            document.getElementById('console2').innerHTML = "Console2 prediction: " + classes[result2.label];

        } catch(error){
            console.log('Modelo aun no configurado')
        }

        document.getElementById('console').innerHTML = 'prediction' 
            + result[0].className + "probability" + result[0].probability

            // elimina imagen de memoria
            img.dispose()

            await tf.nextFrame();
        }
}



imgEl.onload = async function () {
    displayImagePrediction();
}

async function displayImagePrediction() {
    try {
        result = await net.classify(imgEl);
        descEl.innerHTML = JSON.stringify(result);
    } catch (error) {

    }
};

async function addExample(classId) {
    console.log('Added Example')
    const img = await webcam.capture();
    const activation = net.infer(img, true);
    classifier.addExample(activation, classId);

    img.dispose();
}

count = 0;

async function cambiarImagen() {
    count = count + 1
    imgEl.src = "https://picsum.photos/200/300?random=" + count;

}
app();