const bootMessages = [

    "INITIALIZING META-RELIABILITY ENGINE...",
    "LOADING TRANSFORMER WEIGHTS...",
    "CALIBRATING ENTROPY DETECTORS...",
    "CONNECTING FAILURE PIPELINE...",
    "LOADING ADVERSARIAL DATABASE...",
    "SYNCHRONIZING META-CLASSIFIER...",
    "SYSTEM READY"

];

const bootText =
    document.getElementById("boot-text");

const bootScreen =
    document.getElementById("boot-screen");

const mainSystem =
    document.getElementById("main-system");


let bootIndex = 0;

function showBootMessage() {

    if (bootIndex < bootMessages.length) {

        bootText.innerHTML +=
            bootMessages[bootIndex] + "<br><br>";

        bootIndex++;

        setTimeout(showBootMessage, 700);

    }

    else {

        setTimeout(() => {

            bootScreen.style.display = "none";

            mainSystem.style.display = "block";

        }, 1200);

    }

}

if (!sessionStorage.getItem("booted")) {

    showBootMessage();

    sessionStorage.setItem("booted", "true");

}

else {

    bootScreen.style.display = "none";

    mainSystem.style.display = "block";

}


function switchPanel(panelId) {

    const panels =
        document.querySelectorAll(".content-panel");

    panels.forEach(panel => {

        panel.classList.remove("active-panel");

    });

    document
        .getElementById(panelId)
        .classList.add("active-panel");

    const buttons =
        document.querySelectorAll(".menu-btn");

    buttons.forEach(btn => {

        btn.classList.remove("active");

    });

    event.target.classList.add("active");

}


function randomFloat(min, max, decimals = 3) {

    return (
        Math.random() * (max - min) + min
    ).toFixed(decimals);

}


function updateTelemetry() {

    document.getElementById("anomaly")
        .innerText =
        randomFloat(0.200, 0.900);

    document.getElementById("load")
        .innerText =
        Math.floor(Math.random() * 100) + "%";

}

setInterval(updateTelemetry, 1200);


const logContainer =
    document.getElementById("logs");

const logMessages = [

    `entropy level detected: ${backendData.entropy}`,

    `failure probability: ${backendData.failureProbability}`,

    `bert confidence stabilized at ${backendData.bertConfidence}`,

    `cross-model disagreement analysis complete`,

    `meta-classifier confidence updated`,

    `adversarial drift scan complete`,

    `entropy threshold recalibrated`,

    `prediction reliability pipeline active`

];


function generateLog() {

    const div =
        document.createElement("div");

    const now =
        new Date();

    const time =
        now.toLocaleTimeString();

    const randomMessage =
        logMessages[
            Math.floor(
                Math.random() * logMessages.length
            )
        ];

    div.innerText =
        `[${time}] ${randomMessage}`;

    logContainer.prepend(div);

    if (logContainer.children.length > 20) {

        logContainer.removeChild(
            logContainer.lastChild
        );

    }

}

setInterval(generateLog, 1800);


const entropyCtx =
    document.getElementById("entropyChart");

new Chart(entropyCtx, {

    type: "line",

    data: {

        labels: [
            "T1",
            "T2",
            "T3",
            "T4",
            "T5",
            "T6"
        ],

        datasets: [{

            label: "Entropy Drift",

    data: [

    backendData.entropy * 200,

    backendData.entropy * 350,

    backendData.entropy * 500,

    backendData.entropy * 700,

    backendData.entropy * 900,

    backendData.entropy * 600

],

            borderColor: "white",

            borderWidth: 2

        }]

    },

    options: {

        responsive: true,
        maintainAspectRatio: false,

        plugins: {

            legend: {

                labels: {

                    color: "white"

                }

            }

        },

        scales: {

            x: {

                ticks: {

                    color: "white"

                }

            },

            y: {

                ticks: {

                    color: "white"

                }

            }

        }

    }

});


const confidenceCtx =
    document.getElementById("confidenceChart");

new Chart(confidenceCtx, {

    type: "bar",

    data: {

        labels: [

            "VADER",
            "TF-IDF",
            "BERT"

        ],

        datasets: [{

            label: "Confidence",

            data: [

    backendData.vaderConfidence,

    backendData.lrConfidence,

    backendData.bertConfidence

],

            backgroundColor: [

                "rgba(255,255,255,0.3)",
                "rgba(255,255,255,0.5)",
                "rgba(255,255,255,0.8)"

            ],

            borderColor: "white",

            borderWidth: 2

        }]

    },

    options: {

        responsive: true,
        maintainAspectRatio: false,

        plugins: {

            legend: {

                labels: {

                    color: "white"

                }

            }

        },

        scales: {

            x: {

                ticks: {

                    color: "white"

                }

            },

            y: {

                ticks: {

                    color: "white"

                }

            }

        }

    }

});


function flickerEffect() {

    document.body.style.opacity =
        Math.random() > 0.97
            ? "0.96"
            : "1";

}

setInterval(flickerEffect, 100);

const gauge =
    document.querySelector(".risk-gauge");

if (gauge) {

    gauge.style.setProperty(

        "--risk",

        backendData.failureProbability * 100

    );

}
