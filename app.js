/**
 * Airline Passenger Satisfaction Prediction — Application Logic
 * ================================================================
 *
 * This file drives the single-page app defined in index.html. It expects
 * four global constants to already exist, loaded earlier by data.js:
 *
 *   MODEL     The trained Random Forest, exported as a tree structure.
 *   SAMPLES   Forty real passengers, used by the "Load a real passenger" button.
 *   METRICS   Accuracy / precision / recall / F1 / ROC-AUC for each model.
 *   EDA       Exploratory data analysis summaries (segment and rating averages).
 *
 * The file is organized into the following sections:
 *   1. Page navigation
 *   2. Service rating sliders (built once, at startup)
 *   3. Feature encoding (form values -> the numbers the model expects)
 *   4. Random Forest prediction engine
 *   5. Sample passenger loader
 *   6. Results page rendering (prediction, contributions, EDA comparisons)
 *   7. Home page summary cards
 *   8. Models page (accuracy table, confusion matrix)
 *   9. Insights page (feature importance, published evidence)
 *   10. Modal dialog and footer actions
 *   11. Startup
 */


/* ============================================================
   1. Page navigation
   ============================================================ */

/**
 * Show one page of the single-page app and hide the others.
 * Also updates which tab in the top navigation bar looks active.
 * @param {string} pageId - The id of the <section class="page"> to show.
 */
function showPage(pageId) {
  document.querySelectorAll(".page").forEach((section) => {
    section.classList.toggle("on", section.id === pageId);
  });
  document.querySelectorAll(".tab").forEach((tabButton) => {
    tabButton.classList.toggle("on", tabButton.dataset.page === pageId);
  });
  window.scrollTo(0, 0);
}

/**
 * Navigate to a page and remember it in the URL, so the browser's
 * back/forward buttons and page refresh behave sensibly.
 * @param {string} pageId
 */
function navigateTo(pageId) {
  showPage(pageId);
  try {
    history.replaceState(null, "", "#" + pageId);
  } catch (error) {
    // Some embedded/preview environments block history changes.
    // Navigation still works without it, so this is safe to ignore.
  }
}

window.addEventListener("hashchange", () => {
  const requestedPage = location.hash.slice(1);
  const pageExists = document.getElementById(requestedPage) !== null;
  if (pageExists && requestedPage !== "results") {
    showPage(requestedPage);
  }
});


/* ============================================================
   2. Service rating sliders
   ============================================================ */

/**
 * Every service rating field as [inputId, displayLabel, modelColumnName].
 * inputId is used for the HTML element id, displayLabel is shown to the
 * user, and modelColumnName must exactly match a column name inside
 * MODEL.features (see train_model.py) so the encoding step below lines up.
 */
const SERVICE_RATINGS = [
  ["wifi", "Inflight wifi service", "Inflight wifi service"],
  ["timeConv", "Departure/Arrival time convenient", "Departure/Arrival time convenient"],
  ["booking", "Ease of Online booking", "Ease of Online booking"],
  ["gate", "Gate location", "Gate location"],
  ["food", "Food and drink", "Food and drink"],
  ["boarding", "Online boarding", "Online boarding"],
  ["seat", "Seat comfort", "Seat comfort"],
  ["entertainment", "Inflight entertainment", "Inflight entertainment"],
  ["onboard", "On-board service", "On-board service"],
  ["legroom", "Leg room service", "Leg room service"],
  ["baggage", "Baggage handling", "Baggage handling"],
  ["checkin", "Checkin service", "Checkin service"],
  ["inflightService", "Inflight service", "Inflight service"],
  ["clean", "Cleanliness", "Cleanliness"],
];

/** Build the fourteen rating sliders once, into the #ratings-grid container. */
function buildRatingSliders() {
  const container = document.getElementById("ratings-grid");
  SERVICE_RATINGS.forEach(([inputId, label]) => {
    const row = document.createElement("div");
    row.innerHTML = `
      <label>${label}</label>
      <div class="rr">
        <input
          type="range" min="1" max="5" value="3"
          id="${inputId}"
          oninput="document.getElementById('${inputId}-value').textContent = this.value"
        >
        <span class="rv" id="${inputId}-value">3</span>
      </div>`;
    container.appendChild(row);
  });
}


/* ============================================================
   3. Feature encoding
   ============================================================ */

/**
 * Maps each categorical field to the same numeric codes used when the
 * model was trained (see CATEGORY_MAPS in train_model.py). The two
 * must always match, or the model will misread the input.
 */
const CATEGORY_CODES = {
  "Gender": { "Female": 0, "Male": 1 },
  "Customer Type": { "Loyal Customer": 0, "disloyal Customer": 1 },
  "Type of Travel": { "Business travel": 0, "Personal Travel": 1 },
  "Class": { "Business": 0, "Eco": 1, "Eco Plus": 2 },
};

/** Read one form field's current value by its element id. */
function fieldValue(elementId) {
  return document.getElementById(elementId).value;
}

/**
 * Read every form field and assemble them, in the exact order
 * MODEL.features expects, into the numeric vector the forest needs.
 * @returns {number[]}
 */
function buildFeatureVector() {
  const values = {
    "Gender": fieldValue("Gender"),
    "Customer Type": fieldValue("CustomerType"),
    "Age": Number(fieldValue("Age")) || 0,
    "Type of Travel": fieldValue("TravelType"),
    "Class": fieldValue("Class"),
    "Flight Distance": Number(fieldValue("Distance")) || 0,
    "Departure Delay in Minutes": Number(fieldValue("DepDelay")) || 0,
    "Arrival Delay in Minutes": Number(fieldValue("ArrDelay")) || 0,
  };

  SERVICE_RATINGS.forEach(([inputId, , modelColumnName]) => {
    values[modelColumnName] = Number(fieldValue(inputId));
  });

  return MODEL.features.map((columnName) => {
    const categoryMap = CATEGORY_CODES[columnName];
    return categoryMap ? categoryMap[values[columnName]] : values[columnName];
  });
}

/** The inverse of CATEGORY_CODES: turn a numeric code back into its label. */
function decodeCategory(columnName, code) {
  const categoryMap = CATEGORY_CODES[columnName];
  if (!categoryMap) return code;
  return Object.keys(categoryMap).find((label) => categoryMap[label] === code);
}


/* ============================================================
   4. Random Forest prediction engine
   ============================================================ */

/**
 * Walk one decision tree for a single passenger.
 * Each node is [featureIndex, threshold, leftChild, rightChild, probability].
 * A leaf node has featureIndex === -1.
 * @param {Array} treeNodes - One tree from MODEL.trees.
 * @param {number[]} featureVector
 * @returns {number} The probability of "satisfied" at the leaf reached.
 */
function traverseTree(treeNodes, featureVector) {
  let nodeIndex = 0;
  for (;;) {
    const [featureIndex, threshold, leftChild, rightChild] = treeNodes[nodeIndex];
    if (featureIndex === -1) {
      return treeNodes[nodeIndex][4];
    }
    nodeIndex = featureVector[featureIndex] <= threshold ? leftChild : rightChild;
  }
}

/**
 * Average every tree's vote to get the forest's overall probability.
 * @param {number[]} featureVector
 * @returns {number} Probability of "satisfied", between 0 and 1.
 */
function predictSatisfactionProbability(featureVector) {
  const treeCount = MODEL.trees.length;
  const total = MODEL.trees.reduce((sum, tree) => sum + traverseTree(tree, featureVector), 0);
  return total / treeCount;
}

/**
 * Explain a prediction by tracking how much each feature moved the
 * probability as the passenger's data moved down each tree. Averaged
 * across all trees, these contributions add up exactly to the final
 * probability, which is what makes the "Why this prediction" bars honest.
 * @param {number[]} featureVector
 * @returns {{ bias: number, contributions: number[], probability: number }}
 */
function explainPrediction(featureVector) {
  const treeCount = MODEL.trees.length;
  const contributions = new Array(featureVector.length).fill(0);
  let bias = 0;

  MODEL.trees.forEach((treeNodes) => {
    let nodeIndex = 0;
    bias += treeNodes[0][4] / treeCount;

    while (treeNodes[nodeIndex][0] !== -1) {
      const [featureIndex, threshold, leftChild, rightChild] = treeNodes[nodeIndex];
      const nextIndex = featureVector[featureIndex] <= threshold ? leftChild : rightChild;
      const probabilityBefore = treeNodes[nodeIndex][4];
      const probabilityAfter = treeNodes[nextIndex][4];
      contributions[featureIndex] += (probabilityAfter - probabilityBefore) / treeCount;
      nodeIndex = nextIndex;
    }
  });

  const probability = bias + contributions.reduce((sum, value) => sum + value, 0);
  return { bias, contributions, probability };
}


/* ============================================================
   5. Sample passenger loader
   ============================================================ */

/** The true outcome of the most recently loaded sample, or null for a manually-filled form. */
let lastKnownActualOutcome = null;

/** Fill the form with one random real passenger from SAMPLES, then predict. */
function loadRandomPassenger() {
  const passenger = SAMPLES[Math.floor(Math.random() * SAMPLES.length)];

  document.getElementById("Gender").value = passenger["Gender"];
  document.getElementById("CustomerType").value = passenger["Customer Type"];
  document.getElementById("Age").value = passenger["Age"];
  document.getElementById("TravelType").value = passenger["Type of Travel"];
  document.getElementById("Class").value = passenger["Class"];
  document.getElementById("Distance").value = passenger["Flight Distance"];
  document.getElementById("DepDelay").value = passenger["Departure Delay in Minutes"];
  document.getElementById("ArrDelay").value = passenger["Arrival Delay in Minutes"];

  SERVICE_RATINGS.forEach(([inputId, , modelColumnName]) => {
    // The source survey uses 0 for "not applicable"; the slider's
    // minimum is 1, so any 0 rating is shown as 1 instead.
    const ratingValue = Math.max(1, passenger[modelColumnName]);
    document.getElementById(inputId).value = ratingValue;
    document.getElementById(inputId + "-value").textContent = ratingValue;
  });

  lastKnownActualOutcome = passenger.satisfaction;
  runPrediction();
}


/* ============================================================
   6. Results page rendering
   ============================================================ */

/**
 * Run the model on the current form values and render the results page:
 * the headline prediction, the top contributing features, and the two
 * EDA comparison charts.
 */
function runPrediction() {
  const featureVector = buildFeatureVector();
  const { bias, contributions, probability } = explainPrediction(featureVector);

  renderPredictionHeadline(probability);
  renderContributionBars(contributions, featureVector, bias, probability);
  renderRatingComparisonChart();
  renderSegmentComparisonChart();

  navigateTo("results");
}

function renderPredictionHeadline(probability) {
  const isSatisfied = probability >= 0.5;
  const panel = document.getElementById("result-panel");
  const percent = Math.round(probability * 100);

  panel.style.display = "block";
  panel.className = isSatisfied ? "satisfied" : "dissatisfied";

  document.getElementById("result-label").textContent = isSatisfied
    ? "Satisfied"
    : "Neutral or Dissatisfied";
  document.getElementById("result-label").style.color = isSatisfied ? "#1b7a35" : "#a33";
  document.getElementById("result-bar-fill").style.width = percent + "%";
  document.getElementById("result-probability").textContent =
    "Estimated satisfaction likelihood: " + percent + "%";
  document.getElementById("result-actual").textContent = lastKnownActualOutcome
    ? "Actual outcome in dataset: " + lastKnownActualOutcome
    : "";

  document.getElementById("result-panel").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/** Render the "Why this prediction" bars for the eight most influential features. */
function renderContributionBars(contributions, featureVector, bias, probability) {
  document.getElementById("contribution-explainer").textContent =
    `Baseline (average passenger) is ${Math.round(bias * 100)}%. Each bar shows how many ` +
    `percentage points a feature pushed this passenger's likelihood up (right) or down ` +
    `(left). The bars add up to the final ${Math.round(probability * 100)}%.`;

  const rankedFeatures = contributions
    .map((value, index) => ({ name: MODEL.features[index], value, rawValue: featureVector[index] }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, 8);

  const largestMagnitude = Math.max(...rankedFeatures.map((f) => Math.abs(f.value)), 0.01);

  document.getElementById("contribution-bars").innerHTML = rankedFeatures
    .map((feature) => {
      const barWidthPercent = (Math.abs(feature.value) / largestMagnitude) * 50;
      const barSide = feature.value >= 0 ? "left:50%" : "right:50%";
      const barClass = feature.value >= 0 ? "contribution-positive" : "contribution-negative";
      const displayValue = decodeCategory(feature.name, feature.rawValue);
      const signedPoints = (feature.value >= 0 ? "+" : "") + (feature.value * 100).toFixed(1);

      return `
        <div class="contribution-row">
          <span>${feature.name} <span class="note">(${displayValue})</span></span>
          <div class="contribution-axis">
            <i class="${barClass}" style="${barSide};width:${barWidthPercent}%"></i>
          </div>
          <b>${signedPoints}</b>
        </div>`;
    })
    .join("");
}

/** Render the current passenger's ratings against the satisfied/dissatisfied averages. */
function renderRatingComparisonChart() {
  const rows = Object.entries(EDA.means).map(([columnName, [satisfiedAvg, dissatisfiedAvg]]) => {
    const matchingField = SERVICE_RATINGS.find((field) => field[2] === columnName);
    const thisPassengerRating = Number(fieldValue(matchingField[0]));
    const positionPercent = (rating) => (rating / 5) * 100 + "%";

    return `
      <div class="contribution-row" style="grid-template-columns:200px 1fr 30px">
        <span>${columnName}</span>
        <div class="rating-track">
          <b style="left:${positionPercent(dissatisfiedAvg)};background:var(--brown)"></b>
          <b style="left:${positionPercent(satisfiedAvg)};background:#1fb5bb"></b>
          <b style="left:${positionPercent(thisPassengerRating)};background:#111;width:8px"></b>
        </div>
        <b>${thisPassengerRating}</b>
      </div>`;
  });

  document.getElementById("rating-comparison-chart").innerHTML = rows.join("");
}

/** Render satisfaction rate by segment, highlighting this passenger's own segment. */
function renderSegmentComparisonChart() {
  const ageValue = Number(fieldValue("Age"));
  const delayValue = Number(fieldValue("DepDelay"));

  const thisPassengerSegment = {
    "Class": fieldValue("Class"),
    "Type of Travel": fieldValue("TravelType"),
    "Customer Type": fieldValue("CustomerType"),
    "Age band": ageValue <= 20 ? "<=20" : ageValue <= 35 ? "21-35" : ageValue <= 50 ? "36-50" : ageValue <= 65 ? "51-65" : "65+",
    "Delay band": delayValue <= 0 ? "None" : delayValue <= 15 ? "1-15 min" : delayValue <= 60 ? "16-60 min" : ">60 min",
  };

  const groupSections = Object.entries(EDA.seg).map(([groupName, groupValues]) => {
    const bars = Object.entries(groupValues)
      .map(([label, [rate, passengerCount]]) => {
        const isThisPassenger = label === thisPassengerSegment[groupName];
        return `
          <div class="segment-bar${isThisPassenger ? " me" : ""}">
            <span>${label}</span>
            <i style="width:${rate * 100}%"></i>
            <span>${(rate * 100).toFixed(0)}% <span class="note">n=${passengerCount.toLocaleString()}</span></span>
          </div>`;
      })
      .join("");

    return `<div class="segment-group"><h5>${groupName}</h5>${bars}</div>`;
  });

  document.getElementById("segment-comparison-chart").innerHTML = groupSections.join("");
}


/* ============================================================
   7. Home page summary cards
   ============================================================ */

function renderHomeSummaryCards() {
  const deployedModel = METRICS.models.find((model) => model.name.includes("deployed"));
  const topFeatureName = METRICS.imp[0][0];

  document.getElementById("card-total-passengers").textContent = METRICS.n_total.toLocaleString();
  document.getElementById("card-split-ratio").textContent = "80 / 20";
  document.getElementById("card-model-accuracy").textContent = (deployedModel.acc * 100).toFixed(1) + "%";
  document.getElementById("card-top-driver").textContent = topFeatureName;
}


/* ============================================================
   8. Models page
   ============================================================ */

function renderModelsPage() {
  document.getElementById("split-train-label").textContent = "Train 80% \u00b7 " + METRICS.n_train.toLocaleString();
  document.getElementById("split-test-label").textContent = "Test 20%";

  const table = document.getElementById("models-table");
  METRICS.models.forEach((model) => {
    const row = table.insertRow();
    if (model.name.includes("deployed")) row.className = "best";

    const asPercent = (value) => (value * 100).toFixed(1) + "%";
    [model.name, asPercent(model.acc), asPercent(model.prec), asPercent(model.rec), asPercent(model.f1), model.auc.toFixed(3)]
      .forEach((cellValue) => (row.insertCell().textContent = cellValue));
  });

  const [[trueNegative, falsePositive], [falseNegative, truePositive]] = METRICS.cm;
  document.getElementById("confusion-matrix").innerHTML = `
    <div></div><div><b>Pred: not satisfied</b></div><div><b>Pred: satisfied</b></div>
    <div><b>Actual: not</b></div><div class="correct">${trueNegative}</div><div class="wrong">${falsePositive}</div>
    <div><b>Actual: satisfied</b></div><div class="wrong">${falseNegative}</div><div class="correct">${truePositive}</div>`;

  const cvMeanPercent = (METRICS.cv_mean * 100).toFixed(1);
  const cvStdPercent = (METRICS.cv_std * 100).toFixed(1);
  document.getElementById("cross-validation-note").textContent =
    `5-fold cross-validation accuracy: ${cvMeanPercent}% \u00b1 ${cvStdPercent}%, consistent with the single ` +
    `split, so the score is unlikely to be a lucky draw.`;
}


/* ============================================================
   9. Insights page
   ============================================================ */

function renderInsightsPage() {
  const largestImportance = METRICS.imp[0][1];

  document.getElementById("feature-importance-bars").innerHTML = METRICS.imp
    .map(([featureName, importanceScore], rank) => {
      const barWidthPercent = (importanceScore / largestImportance) * 100;
      const isTopFive = rank < 5;
      return `
        <div class="importance-bar${isTopFive ? " top" : ""}">
          <span>${featureName}</span>
          <i style="width:${barWidthPercent}%"></i>
          <span>${(importanceScore * 100).toFixed(1)}%</span>
        </div>`;
    })
    .join("");

  const topFiveShare = METRICS.imp.slice(0, 5).reduce((sum, [, score]) => sum + score, 0);
  document.getElementById("pareto-note").textContent =
    `Top 5 of 22 features (about 23%) carry ${(topFiveShare * 100).toFixed(0)}% of the model's total ` +
    `importance, close to an 80/20 (Pareto) pattern. Importance shows predictive weight, not causation.`;
}


/* ============================================================
   10. Modal dialog and footer actions
   ============================================================ */

function openModal(kind) {
  const body = document.getElementById("modal-body");

  if (kind === "auth") {
    body.innerHTML = `
      <h3>Login or Sign Up</h3>
      <input type="email" placeholder="Email">
      <input type="password" placeholder="Password">
      <button class="pbtn" onclick="confirmAuthDemo()">Continue</button>
      <p class="note" id="auth-demo-note">Demo only: this project has no account backend, nothing is stored.</p>`;
  } else {
    body.innerHTML = `
      <h3>Terms &amp; Conditions</h3>
      <p class="note">This site is an academic machine-learning demonstration. Predictions are estimates
      from a survey dataset and must not be used for real operational or personal decisions.
      No personal data is collected or stored.</p>
      <button class="pbtn" onclick="closeModal()">Close</button>`;
  }

  document.getElementById("modal-overlay").classList.add("on");
}

function closeModal() {
  document.getElementById("modal-overlay").classList.remove("on");
}

function confirmAuthDemo() {
  document.getElementById("auth-demo-note").textContent =
    "Thanks! Accounts are not enabled in this demo build.";
}

function subscribeNewsletter() {
  const emailInput = document.getElementById("newsletter-email");
  emailInput.value = "";
  emailInput.placeholder = "Thanks for subscribing!";
}


/* ============================================================
   11. Startup
   ============================================================ */

function initializeApp() {
  buildRatingSliders();
  renderHomeSummaryCards();
  renderModelsPage();
  renderInsightsPage();
}

document.addEventListener("DOMContentLoaded", initializeApp);
