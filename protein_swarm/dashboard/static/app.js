(function () {
  const runForm = document.getElementById("run-form");
  const runBtn = document.getElementById("run-btn");
  const inProgressContent = document.getElementById("in-progress-content");
  const completedContent = document.getElementById("completed-content");
  const currentSequenceEl = document.getElementById("current-sequence");
  const bestScoreEl = document.getElementById("best-score");
  const rosettaEnergyEl = document.getElementById("rosetta-energy");
  const energyTrendEl = document.getElementById("energy-trend");
  const meanPlddtEl = document.getElementById("mean-plddt");

  let eventSource = null;
  let lastSequence = "";

  function setRunning(running) {
    runBtn.disabled = running;
    runBtn.textContent = running ? "Running…" : "Run";
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
    appliedSet = null;
    inProgressContent.innerHTML = "";
    completedContent.innerHTML = "";

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
          case "run_complete":
            if (eventSource) {
              eventSource.close();
              eventSource = null;
            }
            setRunning(false);
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
