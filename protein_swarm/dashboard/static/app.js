(function () {
  const runForm = document.getElementById("run-form");
  const runBtn = document.getElementById("run-btn");
  const inProgressContent = document.getElementById("in-progress-content");
  const completedContent = document.getElementById("completed-content");
  const inProgressBadge = document.getElementById("in-progress-badge");
  const completedBadge = document.getElementById("completed-badge");
  const currentSequenceEl = document.getElementById("current-sequence");
  const bestScoreEl = document.getElementById("best-score");
  const rosettaEnergyEl = document.getElementById("rosetta-energy");
  const energyTrendEl = document.getElementById("energy-trend");
  const logContent = document.getElementById("log-content");
  const proteinModal = document.getElementById("protein-modal");
  const proteinModalViewer = document.getElementById("protein-modal-viewer");
  const proteinModalError = document.getElementById("protein-modal-error");
  const proteinModalClose = document.getElementById("protein-modal-close");
  const proteinModalSave = document.getElementById("protein-modal-save");
  const viewProteinBtn = document.getElementById("view-protein-btn");

  let eventSource = null;
  let proteinViewerInstance = null;
  let lastSequence = "";
  let currentIteration = 0;
  let completedCount = 0;

  (function initRunControls() {
    var maxIterEl = document.getElementById("max_iterations");
    var plateauEl = document.getElementById("plateau_window");

    function syncCustomDropdown(selectId) {
      var sel = document.getElementById(selectId);
      if (!sel) return;
      var wrap = sel.closest(".custom-dropdown-wrap");
      if (!wrap) return;
      var trigger = wrap.querySelector(".custom-dropdown__trigger");
      var list = wrap.querySelector(".custom-dropdown__list");
      if (!trigger || !list) return;
      var val = sel.value || "";
      trigger.innerHTML = val + ' <span class="custom-dropdown__chevron">\u25BC</span>';
      list.innerHTML = "";
      for (var k = 0; k < sel.options.length; k++) {
        var op = sel.options[k];
        var div = document.createElement("div");
        div.className = "custom-dropdown__option" + (op.selected ? " is-selected" : "");
        div.setAttribute("role", "option");
        div.setAttribute("data-value", op.value);
        div.textContent = op.textContent;
        (function (optionValue) {
          div.addEventListener("click", function () {
            sel.value = optionValue;
            trigger.innerHTML = optionValue + ' <span class="custom-dropdown__chevron">\u25BC</span>';
            list.querySelectorAll(".custom-dropdown__option").forEach(function (el) { el.classList.toggle("is-selected", el.getAttribute("data-value") === optionValue); });
            list.classList.remove("is-open");
            trigger.setAttribute("aria-expanded", "false");
            if (selectId === "max_iterations") fillPlateauWindow();
          });
        })(op.value);
        list.appendChild(div);
      }
    }

    function setupCustomDropdown(selectId) {
      var wrap = document.getElementById(selectId + "_wrap") || document.getElementById(selectId).closest(".custom-dropdown-wrap");
      if (!wrap) return;
      var trigger = wrap.querySelector(".custom-dropdown__trigger");
      var list = wrap.querySelector(".custom-dropdown__list");
      if (!trigger || !list) return;
      trigger.addEventListener("click", function (e) {
        e.preventDefault();
        var isOpen = list.classList.toggle("is-open");
        trigger.setAttribute("aria-expanded", isOpen ? "true" : "false");
        if (isOpen) {
          var opt = list.querySelector("[data-value=\"" + document.getElementById(selectId).value + "\"]");
          if (opt) opt.scrollIntoView({ block: "nearest" });
        }
      });
      document.addEventListener("click", function (e) {
        if (!wrap.contains(e.target)) {
          list.classList.remove("is-open");
          trigger.setAttribute("aria-expanded", "false");
        }
      });
    }

    if (maxIterEl) {
      for (var i = 1; i <= 50; i++) {
        var opt = document.createElement("option");
        opt.value = String(i);
        opt.textContent = String(i);
        if (i === 50) opt.selected = true;
        maxIterEl.appendChild(opt);
      }
      syncCustomDropdown("max_iterations");
      setupCustomDropdown("max_iterations");
    }
    function fillPlateauWindow() {
      if (!plateauEl || !maxIterEl) return;
      var max = parseInt(maxIterEl.value, 10) || 50;
      var current = plateauEl.value;
      var plateauMax = max <= 2 ? 2 : max - 1;
      plateauEl.innerHTML = "";
      for (var j = 2; j <= plateauMax; j++) {
        var o = document.createElement("option");
        o.value = String(j);
        o.textContent = String(j);
        if (String(j) === current) o.selected = true;
        plateauEl.appendChild(o);
      }
      if (plateauEl.selectedIndex < 0 && plateauEl.options.length) {
        var defaultVal = Math.min(10, Math.max(2, plateauMax));
        plateauEl.value = String(defaultVal);
      }
      syncCustomDropdown("plateau_window");
    }
    fillPlateauWindow();
    setupCustomDropdown("plateau_window");
    if (maxIterEl) maxIterEl.addEventListener("change", fillPlateauWindow);

    ["llm_model", "remote_fold_backend", "rosetta_relax_cycles"].forEach(function (id) {
      syncCustomDropdown(id);
      setupCustomDropdown(id);
    });

    var mutationRateEl = document.getElementById("mutation_rate");
    var mutationRateVal = document.getElementById("mutation_rate_value");
    if (mutationRateEl && mutationRateVal) {
      function updateMutationVal() { mutationRateVal.textContent = mutationRateEl.value; }
      mutationRateEl.addEventListener("input", updateMutationVal);
      updateMutationVal();
    }
    var confThreshEl = document.getElementById("confidence_threshold");
    var confThreshVal = document.getElementById("confidence_threshold_value");
    if (confThreshEl && confThreshVal) {
      function updateConfVal() { confThreshVal.textContent = confThreshEl.value; }
      confThreshEl.addEventListener("input", updateConfVal);
      updateConfVal();
    }

    var rosettaRelaxEl = document.getElementById("rosetta_relax");
    var relaxCyclesEl = document.getElementById("rosetta_relax_cycles");
    if (rosettaRelaxEl && relaxCyclesEl) {
      rosettaRelaxEl.addEventListener("change", function () {
        if (!rosettaRelaxEl.checked) {
          relaxCyclesEl.value = "0";
          syncCustomDropdown("rosetta_relax_cycles");
        }
      });
    }
  })();

  const spinnerSvg = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"14\" height=\"14\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\"><line x1=\"12\" y1=\"2\" x2=\"12\" y2=\"6\"/><line x1=\"12\" y1=\"18\" x2=\"12\" y2=\"22\"/><line x1=\"4.93\" y1=\"4.93\" x2=\"7.76\" y2=\"7.76\"/><line x1=\"16.24\" y1=\"16.24\" x2=\"19.07\" y2=\"19.07\"/><line x1=\"2\" y1=\"12\" x2=\"6\" y2=\"12\"/><line x1=\"18\" y1=\"12\" x2=\"22\" y2=\"12\"/><line x1=\"4.93\" y1=\"19.07\" x2=\"7.76\" y2=\"16.24\"/><line x1=\"16.24\" y1=\"7.76\" x2=\"19.07\" y2=\"4.93\"/></svg>";
  const checkSvg = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"14\" height=\"14\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\"><path d=\"M22 11.08V12a10 10 0 1 1-5.93-9.14\"/><polyline points=\"22 4 12 14.01 9 11.01\"/></svg>";

  function escapeHtml(s) {
    const div = document.createElement("div");
    div.textContent = s;
    return div.innerHTML;
  }

  const TERMINAL_PREFIX = "biocol.tech > ";
  const isFinalMessage = function (m) { return m.indexOf("Final Protein Design:") === 0; };

  function appendLog(message, icon, variant) {
    if (!logContent) return;
    if (icon === "check") {
      const entries = logContent.querySelectorAll(".log-entry[data-icon=\"spinner\"]");
      for (let i = entries.length - 1; i >= 0; i--) {
        const el = entries[i];
        if (el.getAttribute("data-message") === message) {
          const iconEl = el.querySelector(".log-entry__icon");
          if (iconEl) {
            iconEl.className = "log-entry__icon log-entry__icon--check";
            iconEl.innerHTML = checkSvg;
            el.setAttribute("data-icon", "check");
          }
          return;
        }
      }
    }
    const iconClass = icon === "spinner" ? "log-entry__icon log-entry__icon--spinner" : icon === "check" ? "log-entry__icon log-entry__icon--check" : "log-entry__icon";
    const iconHtml = icon === "spinner" ? "<span class=\"" + iconClass + "\">" + spinnerSvg + "</span>" : icon === "check" ? "<span class=\"" + iconClass + "\">" + checkSvg + "</span>" : "<span class=\"log-entry__icon\"></span>";
    const prefixHtml = "<span class=\"log-entry__prefix\">" + escapeHtml(TERMINAL_PREFIX) + "</span>";
    var lineContent = "";
    var extraClass = "";
    if (variant === "success") extraClass += " log-entry--success";
    if (variant === "error") extraClass += " log-entry--error";
    if (isFinalMessage(message)) {
      extraClass += " log-entry--final";
      var seq = message.slice("Final Protein Design:".length).trim();
      lineContent = "<span class=\"log-entry__text\">Final Protein Design: <span class=\"log-final-box\">" + escapeHtml(seq) + "</span></span>";
    } else {
      lineContent = "<span class=\"log-entry__text\">" + escapeHtml(message) + "</span>";
    }
    const entry = document.createElement("div");
    entry.className = "log-entry" + extraClass;
    entry.setAttribute("data-message", message);
    entry.setAttribute("data-icon", icon || "none");
    entry.innerHTML = iconHtml + prefixHtml + "<span class=\"log-entry__line\">" + lineContent + "</span>";
    logContent.appendChild(entry);
    logContent.scrollTop = logContent.scrollHeight;
  }

  function setRunning(running) {
    runBtn.disabled = running;
    runBtn.textContent = running ? "Running…" : "Run";
  }

  function trendPillClass(trend) {
    if (!trend) return "blue";
    const t = String(trend).toLowerCase();
    if (t === "improving") return "green";
    if (t === "worsening") return "red";
    if (t === "flat") return "yellow";
    return "blue";
  }

  function fmtScore(v) {
    if (v == null || v === undefined) return "—";
    return Number(v).toFixed(4);
  }

  function updateSequencePanel(data) {
    if (data.current_sequence != null) lastSequence = data.current_sequence;
    currentSequenceEl.textContent = data.current_sequence || "—";
    currentSequenceEl.classList.toggle("empty", !data.current_sequence);
    bestScoreEl.textContent = fmtScore(data.best_score);
    rosettaEnergyEl.textContent = data.rosetta_energy != null ? fmtScore(data.rosetta_energy) : "—";
    energyTrendEl.textContent = data.energy_trend || "—";
    energyTrendEl.className = "status-pill status-pill--" + trendPillClass(data.energy_trend);
  }

  function addInProgress(html, options) {
    const div = document.createElement("div");
    div.className = "agent-entry running" + (options && options.noBorder ? " sub" : "");
    div.innerHTML = html;
    inProgressContent.appendChild(div);
    inProgressContent.scrollTop = inProgressContent.scrollHeight;
    return div;
  }

  function addCompleted(html, accepted) {
    const div = document.createElement("div");
    div.className = "agent-entry " + (accepted ? "complete" : "failed");
    div.innerHTML = html;
    completedContent.appendChild(div);
    completedContent.scrollTop = completedContent.scrollHeight;
    return div;
  }

  function getVal(id, def) {
    var el = document.getElementById(id);
    return el && el.value !== undefined ? el.value : def;
  }
  function getChecked(id, def) {
    var el = document.getElementById(id);
    return el ? !!el.checked : (def !== undefined ? def : false);
  }

  function num(val, defaultVal) {
    var n = Number(val);
    return (n !== n || n === undefined) ? defaultVal : n;
  }

  function buildPayload() {
    var maxIter = Math.max(1, Math.min(500, num(parseInt(getVal("max_iterations", "50"), 10), 50)));
    var plateauRaw = num(parseInt(getVal("plateau_window", "5"), 10), 5);
    var plateau_window = Math.max(2, Math.min(plateauRaw, maxIter - 1));
    return {
      sequence: (getVal("sequence") || "").trim(),
      objective: (getVal("objective") || "").trim(),
      max_iterations: maxIter,
      mutation_rate: Math.max(0, Math.min(1, num(parseFloat(getVal("mutation_rate", "0.3")), 0.3))),
      confidence_threshold: Math.max(0, Math.min(1, num(parseFloat(getVal("confidence_threshold", "0.5")), 0.5))),
      plateau_window: plateau_window,
      output_dir: getVal("output_dir") || "outputs",
      use_llm: getChecked("use_llm"),
      llm_provider: getVal("llm_provider") || "openai",
      llm_model: getVal("llm_model", "gpt-4o-mini"),
      modal_fold: getChecked("modal_fold"),
      modal_parallel: getChecked("modal_parallel", true),
      fold_backend: getVal("fold_backend") || "dummy",
      remote_fold_backend: getVal("remote_fold_backend", "esmfold"),
      use_rosetta: getChecked("use_rosetta"),
      rosetta_relax: getChecked("rosetta_relax"),
      rosetta_relax_cycles: Math.max(0, Math.min(20, num(parseInt(getVal("rosetta_relax_cycles", "0"), 10), 0))),
      debug: true,
      dump_prompts: true,
    };
  }

  function agentAssignmentsHtml(iteration, numAgents) {
    if (!lastSequence || lastSequence.length === 0) {
      return `<div class="assignments">Running ${numAgents} agents…</div>`;
    }
    const rows = [];
    const len = Math.min(numAgents, lastSequence.length);
    for (let i = 0; i < len; i++) {
      const pos = i + 1;
      const res = lastSequence[i] || "?";
      rows.push(`<span class="assign-chip" title="Position ${pos}">${pos}:${res}</span>`);
    }
    if (len < numAgents) {
      for (let i = len; i < numAgents; i++) {
        rows.push(`<span class="assign-chip">${i + 1}:?</span>`);
      }
    }
    return `<div class="assignments">Agent assignments: ${rows.join(" ")}</div>`;
  }

  function proposalsTableHtml(proposals, applied) {
    if (!proposals || proposals.length === 0) {
      return "<p class=\"table-empty\">No proposals.</p>";
    }
    const appliedPositions = new Set((applied || []).map(function (m) { return m.position; }));
    let rows = proposals.map(function (p) {
      const applied = appliedPositions.has(p.position);
      const reason = (p.reason || "").replace(/</g, "&lt;").replace(/>/g, "&gt;");
      return (
        "<tr class=\"" + (applied ? "applied" : "") + "\">" +
        "<td>" + (p.position + 1) + "</td>" +
        "<td><code>" + (p.current_residue || "—") + "</code></td>" +
        "<td><code>" + (p.proposed_residue || "—") + "</code></td>" +
        "<td>" + (p.confidence != null ? Number(p.confidence).toFixed(3) : "—") + "</td>" +
        "<td class=\"reason\">" + (reason || "—") + "</td>" +
        "</tr>"
      );
    }).join("");
    return (
      "<table class=\"proposals-table\">" +
      "<thead><tr><th>Pos</th><th>Current</th><th>Proposed</th><th>Conf</th><th>Reason</th></tr></thead>" +
      "<tbody>" + rows + "</tbody></table>"
    );
  }

  if (runForm) {
  runForm.addEventListener("submit", async function (e) {
    e.preventDefault();
    if (runBtn && runBtn.disabled) return;

    var payload;
    try {
      payload = buildPayload();
    } catch (err) {
      alert("Failed to read form: " + (err && err.message ? err.message : String(err)));
      return;
    }
    if (!payload.sequence || !payload.objective) {
      alert("Please enter sequence and objective.");
      return;
    }

    setRunning(true);
    lastSequence = "";
    currentIteration = 0;
    completedCount = 0;
    if (inProgressBadge) inProgressBadge.textContent = "0";
    if (completedBadge) completedBadge.textContent = "0";
    inProgressContent.innerHTML = "";
    completedContent.innerHTML = "";
    if (logContent) logContent.innerHTML = "";
    var agentContent = document.getElementById("agent-window-content");
    if (agentContent) agentContent.classList.remove("run-finished");

    try {
      const res = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!data.ok) {
        alert(data.error || "Failed to start run.");
        setRunning(false);
        return;
      }
    } catch (err) {
      alert("Request failed: " + err.message);
      setRunning(false);
      return;
    }

    eventSource = new EventSource("/api/events");
    eventSource.onmessage = function (ev) {
      try {
        const event = JSON.parse(ev.data);
        switch (event.type) {
          case "run_start":
            addInProgress(
              "<span class=\"iter\">Run started</span> — length " + event.sequence_length +
              ", max iterations " + event.max_iterations + "."
            );
            break;
          case "iteration_start":
            currentIteration = event.iteration + 1;
            if (inProgressBadge) inProgressBadge.textContent = String(currentIteration);
            addInProgress(
              "<span class=\"iter\">Iteration " + (event.iteration + 1) + "</span> / " + event.max_iterations + " <span class=\"status\">— starting</span>"
            );
            break;
          case "agents_started":
            addInProgress(
              "<span class=\"iter\">Iter " + (event.iteration + 1) + "</span> <span class=\"status\">— running " + event.num_agents + " agents</span>"
            );
            addInProgress(agentAssignmentsHtml(event.iteration, event.num_agents), { noBorder: true });
            break;
          case "agents_completed": {
            const acc = event.accepted ? "ACCEPTED" : "REJECTED";
            const fold = event.fold_result || {};
            const applied = event.applied || [];
            const proposals = event.proposals || [];
            const tableHtml = proposalsTableHtml(proposals, applied);
            const summary =
              "<div class=\"iter-summary\">" +
              "<span class=\"badge " + (event.accepted ? "accepted" : "rejected") + "\">" + acc + "</span> " +
              "best_score=" + fmtScore(event.best_score) + " " +
              "combined=" + fmtScore(fold.combined_score) + " " +
              "rosetta=" + fmtScore(fold.rosetta_total_score) +
              "</div>";
            completedCount++;
            if (completedBadge) completedBadge.textContent = String(completedCount);
            addCompleted(
              "<div class=\"completed-header\">" +
              "<span class=\"iter\">Iteration " + (event.iteration + 1) + "</span> " + summary + "</div>" +
              "<div class=\"proposals-wrap\">" + tableHtml + "</div>",
              event.accepted
            );
            inProgressContent.innerHTML = "";
            break;
          }
          case "sequence_update":
            updateSequencePanel(event);
            break;
          case "log":
            appendLog(event.message || "", event.icon !== undefined ? event.icon : null, event.variant || null);
            break;
          case "run_complete":
            if (eventSource) {
              eventSource.close();
              eventSource = null;
            }
            setRunning(false);
            if (inProgressBadge) inProgressBadge.textContent = "—";
            inProgressContent.innerHTML = "";
            var agentContent = document.getElementById("agent-window-content");
            if (agentContent) agentContent.classList.add("run-finished");
            if (typeof updateTotalProteins === "function") updateTotalProteins();
            break;
          case "run_error":
            if (eventSource) {
              eventSource.close();
              eventSource = null;
            }
            setRunning(false);
            addCompleted("<span class=\"status\">Error</span> " + (event.message || "Unknown error"), false);
            inProgressContent.innerHTML = "";
            var agentContentErr = document.getElementById("agent-window-content");
            if (agentContentErr) agentContentErr.classList.add("run-finished");
            break;
        }
      } catch (_) {}
    };
    eventSource.onerror = function () {
      if (runBtn.disabled && eventSource && eventSource.readyState === EventSource.CLOSED) return;
      eventSource.close();
      eventSource = null;
      setRunning(false);
    };
  });
  }

  function openProteinModal() {
    if (!proteinModal || !proteinModalViewer || !proteinModalError) return;
    proteinModalError.style.display = "none";
    proteinModalError.textContent = "";
    proteinModalViewer.innerHTML = "";
    proteinViewerInstance = null;
    proteinModal.setAttribute("aria-hidden", "false");

    fetch("/api/final-pdb")
      .then(function (res) {
        if (!res.ok) return res.text().then(function (t) { throw new Error(t || "Failed to load PDB"); });
        return res.text();
      })
      .then(function (pdbText) {
        if (typeof $3Dmol === "undefined") {
          throw new Error("3Dmol.js not loaded");
        }
        function initViewer() {
          var w = proteinModalViewer.offsetWidth;
          var h = proteinModalViewer.offsetHeight;
          if (w <= 0 || h <= 0) {
            setTimeout(initViewer, 50);
            return;
          }
          proteinModalViewer.style.width = w + "px";
          proteinModalViewer.style.height = h + "px";
          var viewer = $3Dmol.createViewer(proteinModalViewer, { backgroundColor: "white" });
          viewer.addModel(pdbText, "pdb");
          viewer.setStyle({ cartoon: { color: "spectrum" } });
          viewer.zoomTo();
          viewer.render();
          proteinViewerInstance = viewer;
        }
        requestAnimationFrame(function () {
          requestAnimationFrame(initViewer);
        });
      })
      .catch(function (err) {
        proteinModalError.textContent = err.message || "Could not load structure.";
        proteinModalError.style.display = "block";
      });
  }

  function closeProteinModal() {
    if (proteinModal) proteinModal.setAttribute("aria-hidden", "true");
    proteinViewerInstance = null;
    if (proteinModalViewer) proteinModalViewer.innerHTML = "";
  }

  if (viewProteinBtn) {
    viewProteinBtn.addEventListener("click", function (e) {
      e.preventDefault();
      openProteinModal();
    });
  }
  if (proteinModalClose) {
    proteinModalClose.addEventListener("click", closeProteinModal);
  }
  if (proteinModal && proteinModal.querySelector(".protein-modal__backdrop")) {
    proteinModal.querySelector(".protein-modal__backdrop").addEventListener("click", closeProteinModal);
  }
  if (proteinModalSave) {
    proteinModalSave.addEventListener("click", function () {
      if (!proteinViewerInstance) return;
      var png = proteinViewerInstance.pngURI();
      var p = typeof png === "object" && png && typeof png.then === "function" ? png : Promise.resolve(png);
      p.then(function (dataUrl) {
        var a = document.createElement("a");
        a.href = dataUrl;
        a.download = "protein_structure.png";
        a.click();
      }).catch(function () {
        if (proteinModalError) {
          proteinModalError.textContent = "Could not export image.";
          proteinModalError.style.display = "block";
        }
      });
    });
  }

  function updateTotalProteins() {
    var el = document.getElementById("total-proteins-count");
    if (!el) return;
    fetch("/api/stats").then(function (r) { return r.json(); }).then(function (d) {
      el.textContent = d.total_proteins != null ? String(d.total_proteins) : "0";
    }).catch(function () { el.textContent = "0"; });
  }

  var runImageModal = document.getElementById("run-image-modal");
  var runImageImg = document.getElementById("run-image-img");
  var runImageSaveBtn = document.getElementById("run-image-save-btn");
  var runImageCloseBtn = document.getElementById("run-image-close-btn");

  function openRunImageModal(runId) {
    if (!runImageModal || !runImageImg) return;
    runImageImg.src = "/api/run-image/" + runId;
    runImageModal.setAttribute("aria-hidden", "false");
  }

  function closeRunImageModal() {
    if (runImageModal) runImageModal.setAttribute("aria-hidden", "true");
    if (runImageImg) runImageImg.src = "";
  }

  if (runImageCloseBtn) runImageCloseBtn.addEventListener("click", closeRunImageModal);
  if (runImageModal && runImageModal.querySelector(".protein-modal__backdrop")) {
    runImageModal.querySelector(".protein-modal__backdrop").addEventListener("click", closeRunImageModal);
  }
  if (runImageSaveBtn) {
    runImageSaveBtn.addEventListener("click", function () {
      var src = runImageImg && runImageImg.src;
      if (!src) return;
      fetch(src).then(function (r) { return r.blob(); }).then(function (blob) {
        var a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = "run-structure.png";
        a.click();
        URL.revokeObjectURL(a.href);
      }).catch(function () {});
    });
  }

  var eyeSvg = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"18\" height=\"18\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\"><path d=\"M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z\"/><circle cx=\"12\" cy=\"12\" r=\"3\"/></svg>";

  function loadHistory() {
    var tbody = document.getElementById("history-tbody");
    var wrap = document.querySelector(".history-table-wrap");
    var emptyEl = document.getElementById("history-empty");
    if (!tbody) return;
    fetch("/api/history").then(function (r) { return r.json(); }).then(function (d) {
      var runs = d.runs || [];
      tbody.innerHTML = "";
      runs.forEach(function (run) {
        var tr = document.createElement("tr");
        tr.setAttribute("data-run-id", String(run.id));
        tr.setAttribute("data-has-image", run.has_image ? "1" : "0");
        var date = run.created_at ? run.created_at.replace("T", " ").slice(0, 19) : "—";
        var start = (run.initial_sequence || "").length > 30 ? (run.initial_sequence || "").slice(0, 30) + "…" : (run.initial_sequence || "—");
        var end = (run.final_sequence || "").length > 30 ? (run.final_sequence || "").slice(0, 30) + "…" : (run.final_sequence || "—");
        var viewCell = document.createElement("td");
        var eyeBtn = document.createElement("button");
        eyeBtn.type = "button";
        eyeBtn.className = "history-eye-btn" + (run.has_image ? "" : " history-eye-btn--no-image");
        eyeBtn.title = run.has_image ? "View 3D structure" : "No 3D image for this run yet";
        eyeBtn.innerHTML = eyeSvg;
        viewCell.appendChild(eyeBtn);
        tr.innerHTML = "<td>" + date + "</td><td><code>" + escapeHtml(start) + "</code></td><td><code>" + escapeHtml(end) + "</code></td><td>" + (run.use_llm ? "Yes" : "No") + "</td><td>" + (run.iterations != null ? run.iterations : "—") + "</td><td>" + (run.protein_length != null ? run.protein_length : "—") + "</td>";
        tr.appendChild(viewCell);
        tbody.appendChild(tr);
      });
      if (wrap) wrap.classList.toggle("has-rows", runs.length > 0);
      if (emptyEl) emptyEl.style.display = runs.length > 0 ? "none" : "block";
    }).catch(function () {
      if (wrap) wrap.classList.remove("has-rows");
      if (emptyEl) emptyEl.style.display = "block";
    });
  }

  (function initHistoryEyeClicks() {
    var tbody = document.getElementById("history-tbody");
    if (!tbody) return;
    tbody.addEventListener("click", function (e) {
      var btn = e.target && e.target.closest && e.target.closest(".history-eye-btn");
      if (!btn) return;
      var row = btn.closest("tr");
      if (!row) return;
      var runId = row.getAttribute("data-run-id");
      var hasImage = row.getAttribute("data-has-image") === "1";
      if (hasImage && runId) openRunImageModal(runId);
    });
  })();

  function renderAboutMarkdown() {
    var scriptEl = document.getElementById("about-markdown");
    var contentEl = document.getElementById("about-content");
    if (!scriptEl || !contentEl) return;
    var raw = scriptEl.textContent || "";
    if (typeof marked !== "undefined") {
      contentEl.innerHTML = (marked.parse ? marked.parse(raw) : marked(raw));
    } else {
      contentEl.textContent = raw;
    }
  }

  function renderContactMarkdown() {
    var scriptEl = document.getElementById("contact-markdown");
    var contentEl = document.getElementById("contact-content");
    if (!scriptEl || !contentEl) return;
    var raw = scriptEl.textContent || "";
    if (typeof marked !== "undefined") {
      contentEl.innerHTML = (marked.parse ? marked.parse(raw) : marked(raw));
    } else {
      contentEl.textContent = raw;
    }
    contentEl.querySelectorAll('a[href^="mailto:"]').forEach(function (a) {
      var span = document.createElement("span");
      span.className = "contact-email";
      span.textContent = a.textContent;
      a.parentNode.replaceChild(span, a);
    });
  }

  document.querySelectorAll(".main-tab").forEach(function (tab) {
    tab.addEventListener("click", function () {
      var tabId = tab.getAttribute("data-tab");
      document.querySelectorAll(".main-tab").forEach(function (t) { t.classList.remove("main-tab--active"); t.setAttribute("aria-selected", "false"); });
      tab.classList.add("main-tab--active");
      tab.setAttribute("aria-selected", "true");
      document.querySelectorAll(".tab-panel").forEach(function (p) { p.classList.remove("tab-panel--active"); p.hidden = true; });
      var panel = document.getElementById("tab-" + tabId);
      if (panel) { panel.classList.add("tab-panel--active"); panel.hidden = false; }
      if (tabId === "history") loadHistory();
      if (tabId === "about") renderAboutMarkdown();
      if (tabId === "contact") renderContactMarkdown();
    });
  });

  var historyRefreshBtn = document.getElementById("history-refresh-btn");
  if (historyRefreshBtn) historyRefreshBtn.addEventListener("click", loadHistory);

  var splashEl = document.getElementById("splash-overlay");
  if (splashEl) {
    splashEl.classList.add("splash-visible");
    setTimeout(function () {
      splashEl.classList.add("splash-fade-out");
      splashEl.style.transition = "opacity 1s ease-out";
      setTimeout(function () {
        splashEl.setAttribute("aria-hidden", "true");
      }, 1000);
    }, 1000);
  }

  updateTotalProteins();
})();
