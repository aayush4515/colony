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
  const meanPlddtEl = document.getElementById("mean-plddt");
  const logContent = document.getElementById("log-content");

  let eventSource = null;
  let lastSequence = "";
  let currentIteration = 0;
  let completedCount = 0;

  const spinnerSvg = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"14\" height=\"14\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\"><line x1=\"12\" y1=\"2\" x2=\"12\" y2=\"6\"/><line x1=\"12\" y1=\"18\" x2=\"12\" y2=\"22\"/><line x1=\"4.93\" y1=\"4.93\" x2=\"7.76\" y2=\"7.76\"/><line x1=\"16.24\" y1=\"16.24\" x2=\"19.07\" y2=\"19.07\"/><line x1=\"2\" y1=\"12\" x2=\"6\" y2=\"12\"/><line x1=\"18\" y1=\"12\" x2=\"22\" y2=\"12\"/><line x1=\"4.93\" y1=\"19.07\" x2=\"7.76\" y2=\"16.24\"/><line x1=\"16.24\" y1=\"7.76\" x2=\"19.07\" y2=\"4.93\"/></svg>";
  const checkSvg = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"14\" height=\"14\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\"><path d=\"M22 11.08V12a10 10 0 1 1-5.93-9.14\"/><polyline points=\"22 4 12 14.01 9 11.01\"/></svg>";

  function escapeHtml(s) {
    const div = document.createElement("div");
    div.textContent = s;
    return div.innerHTML;
  }

  const TERMINAL_PREFIX = "colony.tech > ";
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
      lineContent = "<span class=\"log-entry__text\">Final Protein Design:<br><span class=\"log-final-box\">" + escapeHtml(seq) + "</span></span>";
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

  function fmtPlddt(v) {
    if (v == null || v === undefined) return "—";
    return Number(v).toFixed(1);
  }

  function updateSequencePanel(data) {
    if (data.current_sequence != null) lastSequence = data.current_sequence;
    currentSequenceEl.textContent = data.current_sequence || "—";
    currentSequenceEl.classList.toggle("empty", !data.current_sequence);
    bestScoreEl.textContent = fmtScore(data.best_score);
    rosettaEnergyEl.textContent = data.rosetta_energy != null ? fmtScore(data.rosetta_energy) : "—";
    energyTrendEl.textContent = data.energy_trend || "—";
    energyTrendEl.className = "status-pill status-pill--" + trendPillClass(data.energy_trend);
    meanPlddtEl.textContent = fmtPlddt(data.mean_plddt);
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

  function buildPayload() {
    return {
      sequence: document.getElementById("sequence").value.trim(),
      objective: document.getElementById("objective").value.trim(),
      max_iterations: parseInt(document.getElementById("max_iterations").value, 10),
      mutation_rate: parseFloat(document.getElementById("mutation_rate").value) || 0.3,
      confidence_threshold: parseFloat(document.getElementById("confidence_threshold").value) || 0.5,
      plateau_window: parseInt(document.getElementById("plateau_window").value, 10) || 5,
      use_llm: document.getElementById("use_llm").checked,
      llm_model: document.getElementById("llm_model").value,
      modal_fold: document.getElementById("modal_fold").checked,
      remote_fold_backend: document.getElementById("remote_fold_backend").value,
      use_rosetta: document.getElementById("use_rosetta").checked,
      rosetta_relax: document.getElementById("rosetta_relax").checked,
      rosetta_relax_cycles: parseInt(document.getElementById("rosetta_relax_cycles").value, 10),
      rosetta_norm_target: parseFloat(document.getElementById("rosetta_norm_target").value) || -200,
      rosetta_norm_scale: parseFloat(document.getElementById("rosetta_norm_scale").value) || 50,
      w_physics: parseFloat(document.getElementById("w_physics").value) || 0.55,
      w_objective: parseFloat(document.getElementById("w_objective").value) || 0.35,
      w_confidence: parseFloat(document.getElementById("w_confidence").value) || 0.10,
      debug: document.getElementById("debug").checked,
      dump_prompts: document.getElementById("dump_prompts").checked,
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

  runForm.addEventListener("submit", async function (e) {
    e.preventDefault();
    if (runBtn.disabled) return;

    const payload = buildPayload();
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
            addInProgress(
              "<span class=\"iter\">Run complete.</span> Final length " + (event.final_sequence ? event.final_sequence.length : 0) +
              ", total iterations " + event.total_iterations + "."
            );
            break;
          case "run_error":
            if (eventSource) {
              eventSource.close();
              eventSource = null;
            }
            setRunning(false);
            addCompleted("<span class=\"status\">Error</span> " + (event.message || "Unknown error"), false);
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
})();
