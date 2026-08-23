/**
 * Melodict Standalone Markov Engine & LBDM Segmenter for Max/MSP (WITH DIAGNOSTIC LOGS).
 */

inlets = 1;
outlets = 2; // Outlet 0: Rhythmic MIDI note playback [pitch, duration, velocity], Outlet 1: Boundary bangs

var markovDictionary = {};
var currentPhrase = [];
var maxPhraseLength = 12;
var silenceThresholdMs = 500;
var playbackTask = null;
var lastNoteTime = 0;

function quantizeDuration(durationMs) {
    if (durationMs < 160) return 125;      
    if (durationMs < 350) return 250;      
    if (durationMs < 700) return 500;      
    if (durationMs < 1400) return 1000;    
    return 2000;                           
}

function quantizeVelocity(velocity) {
    if (velocity < 30) return 25;   
    if (velocity < 55) return 50;   
    if (velocity < 85) return 75;   
    if (velocity < 110) return 100; 
    return 127;                     
}

function createTupleState(pitch, duration, velocity) {
    var qDur = quantizeDuration(duration);
    var qVel = quantizeVelocity(velocity);
    return pitch + "_" + qDur + "_" + qVel;
}

function getPhraseStrings(phraseObjArray) {
    var arr = [];
    for (var i = 0; i < phraseObjArray.length; i++) {
        var o = phraseObjArray[i];
        arr.push(createTupleState(o.pitch, o.dur, o.vel));
    }
    return arr;
}

function updateLastNoteDuration(realDur) {
    if (currentPhrase.length === 0) return;
    currentPhrase[currentPhrase.length - 1].dur = realDur;
}

function add_note(pitch, dummyDur, velocity) {
    post("[JS LOG] --> add_note received: Pitch=" + pitch + " | Vel=" + velocity + "\n");

    if (pitch < 21 || pitch > 108 || velocity <= 0) {
        post("[JS LOG] REJECTED: Out of piano range (21-108) or volume 0.\n");
        return;
    }
    
    var now = new Date().getTime();

    
    if (currentPhrase.length > 0 && lastNoteTime > 0) {
        var elapsed = now - lastNoteTime;
        
        
        if (elapsed >= silenceThresholdMs) {
            post("[JS LOG] CUT DUE TO ACOUSTIC SILENCE (" + elapsed + " ms)\n");
            updateLastNoteDuration(500);
            outlet(1, "bang");
            learnPhrase(currentPhrase);
            triggerContinuation(currentPhrase);
            currentPhrase = [];
        } else {
            updateLastNoteDuration(elapsed);
        }
    }

    lastNoteTime = now;

    
    currentPhrase.push({ pitch: pitch, dur: 250, vel: velocity });
    post("[JS LOG] Note #" + currentPhrase.length + " added to buffer: Pitch " + pitch + "\n");

    
    var isBoundary = false;
    if (currentPhrase.length >= 2) {
        var prevPitch = currentPhrase[currentPhrase.length - 2].pitch;
        if (Math.abs(pitch - prevPitch) >= 9) {
            post("[JS LOG] CUT DUE TO MELODIC JUMP (from " + prevPitch + " to " + pitch + ")\n");
            isBoundary = true;
        }
    }
    
    if (currentPhrase.length >= maxPhraseLength) {
        post("[JS LOG] CUT DUE TO MEMORY LIMIT (" + maxPhraseLength + " notes)\n");
        isBoundary = true;
    }

    if (isBoundary && currentPhrase.length >= 1) {
        outlet(1, "bang");
        learnPhrase(currentPhrase);
        triggerContinuation(currentPhrase);
        currentPhrase = [];
    }
}

function learnPhrase(phraseObjs) {
    if (phraseObjs.length < 1) return;
    var phrase = getPhraseStrings(phraseObjs);
    for (var i = 0; i < phrase.length - 1; i++) {
        var currentState = phrase[i];
        var nextState = phrase[i + 1];

        if (!markovDictionary[currentState]) {
            markovDictionary[currentState] = [];
        }
        markovDictionary[currentState].push(nextState);
    }
    post("[JS LOG] Phrase learned with " + phrase.length + " notes! Unique states in memory: " + Object.keys(markovDictionary).length + "\n");
}

function triggerContinuation(phraseObjs) {
    if (phraseObjs.length < 1) return;
    var phrase = getPhraseStrings(phraseObjs);
    var responsePhrase = [];
    var lastState = phrase[phrase.length - 1];
    var targetLength = Math.min(phrase.length, 8);
    post("[JS LOG] Generating " + targetLength + " notes response starting from: " + lastState + "...\n");

    var currentState = lastState;
    for (var step = 0; step < targetLength; step++) {
        var possibleTransitions = markovDictionary[currentState];

        if (!possibleTransitions || possibleTransitions.length === 0) {
            var allStates = Object.keys(markovDictionary);
            if (allStates.length === 0) break;
            currentState = allStates[Math.floor(Math.random() * allStates.length)];
        } else {
            var randomIndex = Math.floor(Math.random() * possibleTransitions.length);
            currentState = possibleTransitions[randomIndex];
        }
        responsePhrase.push(currentState);
    }

    if (responsePhrase.length > 0) {
        post("[JS LOG] Playing generated melody: " + responsePhrase.join(" -> ") + "\n");
        schedulePlayback(responsePhrase);
    } else {
        post("[JS LOG] Could not generate any response sequence.\n");
    }
}

function schedulePlayback(phraseArray) {
    if (playbackTask && playbackTask.running) {
        playbackTask.cancel();
    }

    var stepIndex = 0;
    playbackTask = new Task(function() {
        if (stepIndex >= phraseArray.length) {
            post("[JS LOG] AI playback finished.\n");
            arguments.callee.task.cancel();
            return;
        }

        var parts = phraseArray[stepIndex].split("_");
        var pitch = parseInt(parts[0]);
        var duration = parseInt(parts[1]);
        var velocity = parseInt(parts[2]);

        outlet(0, [pitch, duration, velocity]);
        arguments.callee.task.interval = duration;
        stepIndex++;
    });

    playbackTask.interval = 10;
    playbackTask.repeat();
}

function clear_dictionary() {
    markovDictionary = {};
    currentPhrase = [];
    post("[JS LOG] Markov dictionary reset.\n");
}

function force_boundary() {
    if (currentPhrase.length >= 1) {
        post("[JS LOG] force_boundary() called from Max! Phrase finished with " + currentPhrase.length + " notes.\n");
        updateLastNoteDuration(600); 
        outlet(1, "bang");
        learnPhrase(currentPhrase);
        triggerContinuation(currentPhrase);
        currentPhrase = [];
    }
    
}