const slider = document.getElementById("fps-slider");

const fpsValue = document.getElementById("fps-value");

slider.oninput = function () {

    fpsValue.innerHTML =
        "Alert Seconds: " + this.value;

};