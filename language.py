"""
language.py — which language JARVIS hears and speaks.

One setting, `JARVIS_LANGUAGE`, read here and by nothing else. English is the
base and is not described in this file at all: every English sentence,
cancel word and trigger phrase stays exactly where it was written, beside the
comment that explains it. A second language is a set of OVERRIDES for those,
and a caller asks for one like this:

    speech.say(_t("Cancelled, sir."))                 # a line he speaks
    lang.cancel_words or CANCEL_WORDS                 # a table he matches on

so in English nothing here changes a single byte of what JARVIS says or
matches, and a line nobody has translated yet falls back to English rather
than to silence.

What the setting moves, all of it:

  * the recogniser — `recognition` is the tag the browser's SpeechRecognition
    is given, sent to the page in the `config` frame;
  * the brain — `brain_directive` is appended to the launch prompt, so it is
    rebuilt for every generation and cannot go stale in an edited persona;
  * the voice — `voice_id` is the Fish Audio model used when `FISH_VOICE_ID`
    is not set;
  * every line `server.py`, `speech.py` and `preflight.py` speak themselves,
    without the brain — `lines`;
  * the words matched against what the user said — cancel words, "say that
    again", "start fresh", and the recogniser's known mis-hearings.

What it deliberately does NOT move: the sentences tools hand back to the
brain. Those are read by the model, never spoken, and the directive tells it
to report them in the user's language — translating two hundred of them here
would be two hundred places for the two languages to drift apart. Session
voice names ("the newer hammer") stay as they are for the same reason and a
sharper one: they are identifiers the resolver matches on.
"""
from __future__ import annotations

import logging
import os
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Mapping, Optional

log = logging.getLogger("jarvis.language")

ENV_VAR = "JARVIS_LANGUAGE"


def fold(text: str) -> str:
    """`text` with its accents removed: "detén" -> "deten", "señor" -> "senor".

    Everything matched against what the recogniser returned goes through
    this, on both sides. The recogniser is not consistent about accents —
    "sí" and "si", "está" and "esta" both come back for the same sound — and
    an echo check that compared them exactly would let his own sentence
    through as the user's. ASCII is untouched, so English is unaffected.
    """
    decomposed = unicodedata.normalize("NFD", text or "")
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


@dataclass(frozen=True, eq=False)
class Language:
    """One language JARVIS can run in. Every field but the first two is an
    override; left empty, the caller's own English stands."""
    code: str                     # what JARVIS_LANGUAGE holds, canonically
    recognition: str              # BCP-47 tag for the browser's recogniser
    voice_id: Optional[str] = None
    brain_directive: str = ""
    lines: Mapping[str, str] = field(default_factory=dict)
    cancel_words: tuple = ()
    replay_filler: frozenset = frozenset()
    replay_triggers: frozenset = frozenset()
    fresh_start_phrases: tuple = ()
    abbreviations: frozenset = frozenset()
    number_words: Mapping[int, str] = field(default_factory=dict)
    stt_corrections: Mapping[str, str] = field(default_factory=dict)
    greeting: Optional[Callable[[int], str]] = None
    listing: Optional[Callable[[list], str]] = None
    reset_time: Optional[Callable[[datetime, datetime], str]] = None


# ---------------------------------------------------------------------------
# Mexican Spanish
# ---------------------------------------------------------------------------
#
# Register: the Latin American dub's JARVIS. "Señor" where the English says
# "sir", usted and never tú, and the same economy — a status line is still a
# status line.

_ES_MX_LINES = {
    # speech.py — the mouth's own words
    "As I was saying —": "Como le decía —",
    "Back to it —": "Retomando —",
    "Where was I —": "¿En qué estaba? —",
    "Before I forget —": "Antes de que se me olvide —",
    "My voice is failing, sir.": "Me está fallando la voz, señor.",

    # server.py — the brain's health
    "Claude Code's login has expired, sir — run `claude` in a terminal and "
    "log in, then restart me.":
        "La sesión de Claude Code expiró, señor — ejecute `claude` en una "
        "terminal, inicie sesión y después reiníceme.",
    "Rebooting my language systems, one moment.":
        "Reiniciando mis sistemas de lenguaje, un momento.",
    "My language systems are down, sir. Check the server log.":
        "Mis sistemas de lenguaje están caídos, señor. Revise el registro "
        "del servidor.",
    "My language systems are down, sir.":
        "Mis sistemas de lenguaje están caídos, señor.",
    "One moment, sir — my language systems are still starting.":
        "Un momento, señor — mis sistemas de lenguaje siguen arrancando.",
    "My language systems returned an error, sir. Check the server log.":
        "Mis sistemas de lenguaje devolvieron un error, señor. Revise el "
        "registro del servidor.",
    "I lost my train of thought, sir. Say that again?":
        "Perdí el hilo, señor. ¿Me lo repite?",
    "I've hit the usage limit until {when}, sir.":
        "Alcancé el límite de uso hasta {when}, señor.",
    "later": "más tarde",

    # server.py — the conversation itself
    "I'm afraid I've nothing to repeat yet, sir.":
        "Me temo que aún no tengo nada que repetir, señor.",
    "Gathering my thoughts — one moment, sir.":
        "Ordenando mis ideas — un momento, señor.",
    "Cleared, sir — nothing of that conversation left. Go ahead.":
        "Listo, señor — no queda nada de esa conversación. Adelante.",
    "I couldn't clear it, sir.": "No pude borrarla, señor.",
    "Testing audio, sir.": "Probando audio, señor.",

    # server.py — other sessions
    "A session": "Una sesión",
    "that session": "esa sesión",
    "{name} is {reason}, sir — that one needs your own keystroke.":
        "{name} está {reason}, señor — esa necesita que usted presione la "
        "tecla.",
    "{name} is {reason}, sir.": "{name} está {reason}, señor.",
    "{name} has stopped and wants you, sir.":
        "{name} se detuvo y lo necesita, señor.",
    "waiting on a permission prompt": "esperando un permiso",
    "waiting on a dialog": "esperando respuesta a un diálogo",
    "waiting on input": "esperando su respuesta",
    "waiting on something I cannot name": "esperando algo que no sé nombrar",
    "waiting on {what}": "esperando {what}",
    "{name} has finished, sir.": "{name} terminó, señor.",
    "{count} conversations have finished, sir: {names}.":
        "{count} conversaciones terminaron, señor: {names}.",

    # server.py — his own runs
    "The work in {project} is done, sir.":
        "El trabajo en {project} quedó listo, señor.",
    "Work in {projects} is done, sir.":
        "El trabajo en {projects} quedó listo, señor.",
    "The work in {project} stopped to ask a question, sir, so nothing was "
    "built.":
        "El trabajo en {project} se detuvo para hacer una pregunta, señor, "
        "así que no se construyó nada.",
    "The work in {project} finished, sir, but I can't see that it changed "
    "anything.":
        "El trabajo en {project} terminó, señor, pero no veo que haya "
        "cambiado nada.",
    "The work in {project} ran out of time, sir.":
        "Al trabajo en {project} se le acabó el tiempo, señor.",
    "The work in {project} failed, sir.":
        "El trabajo en {project} falló, señor.",

    # server.py — read-backs: a steer, a command, a keypress
    "Telling {name}: {prompt}": "Le digo a {name}: {prompt}",
    "Cancelled, sir.": "Cancelado, señor.",
    "Passed to {name}, sir.": "Se lo pasé a {name}, señor.",
    " It'll ask you to approve it first — say the word and I'll set your "
    "sessions to accept them.":
        " Primero le pedirá que lo apruebe — dígame y configuro sus "
        "sesiones para que los acepten.",
    "{name} didn't answer its socket, sir — it may have just exited.":
        "{name} no respondió, señor — puede que acabe de cerrarse.",
    "I couldn't deliver that to {name}, sir.":
        "No pude entregárselo a {name}, señor.",
    "Running {command} in {project}, sir.":
        "Voy a ejecutar {command} en {project}, señor.",
    " That isn't a command the project documents, mind.":
        " Ojo: ese comando no está en la documentación del proyecto.",
    "Running in a Terminal window, sir.":
        "Ejecutándose en una ventana de Terminal, señor.",
    "Terminal wouldn't open, sir.": "Terminal no quiso abrirse, señor.",
    "Return": "Enter",
    "Escape": "Escape",
    "Pressing {key} on {name} — this will bring that window forward.":
        "Voy a presionar {key} en {name} — esto traerá esa ventana al "
        "frente.",
    "Pressed {key} on {name}.": "Presioné {key} en {name}.",
    "{name} isn't in a Terminal window I can reach, sir — another "
    "application is hosting it, so that one needs your own hand.":
        "{name} no está en una ventana de Terminal a mi alcance, señor — la "
        "aloja otra aplicación, así que esa le toca a usted.",
    "macOS won't let me send keystrokes, sir — I'd need accessibility "
    "permission in System Settings.":
        "macOS no me deja enviar teclas, señor — necesitaría el permiso de "
        "Accesibilidad en Configuración del Sistema.",
    "{name} has no terminal of its own any more, sir — I pressed nothing.":
        "{name} ya no tiene terminal propia, señor — no presioné nada.",
    "I couldn't press that for {name}, sir.":
        "No pude presionar eso en {name}, señor.",

    # preflight.py — what is wrong at startup
    "One thing needs attention, sir: {issue}.":
        "Una cosa requiere atención, señor: {issue}.",
    "Two things need attention, sir: {first}, and {second}.":
        "Dos cosas requieren atención, señor: {first} y {second}.",
    "{count} things need attention, sir: {issues}, and {last}.":
        "{count} cosas requieren atención, señor: {issues} y {last}.",
    "Claude Code isn't installed": "Claude Code no está instalado",
    "Claude Code needs updating": "Claude Code necesita actualizarse",
    "Claude Code's version couldn't be checked":
        "no se pudo comprobar la versión de Claude Code",
    "Claude Code isn't logged in": "Claude Code no ha iniciado sesión",
    "Claude Code's login couldn't be checked":
        "no se pudo comprobar la sesión de Claude Code",
    "I don't have Accessibility permission":
        "no tengo el permiso de Accesibilidad",
    "Accessibility couldn't be checked":
        "no se pudo comprobar el permiso de Accesibilidad",
    "I don't have Screen Recording permission":
        "no tengo el permiso de Grabación de Pantalla",
    "Screen Recording couldn't be checked":
        "no se pudo comprobar el permiso de Grabación de Pantalla",
    "I have no Fish Audio key": "no tengo la clave de Fish Audio",
    "there's a leftover Anthropic API key in the environment":
        "quedó una clave de API de Anthropic en el entorno",
    "cross-session steering isn't enabled":
        "el envío de mensajes entre sesiones no está activado",
}

_ES_NUMBER_WORDS = {
    1: "uno", 2: "dos", 3: "tres", 4: "cuatro", 5: "cinco",
    6: "seis", 7: "siete", 8: "ocho", 9: "nueve",
}
_ES_WEEKDAYS = ("lunes", "martes", "miércoles", "jueves", "viernes",
                "sábado", "domingo")
_ES_MONTHS = ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
              "agosto", "septiembre", "octubre", "noviembre", "diciembre")


def _es_join(items: list) -> str:
    """'a', 'a y b', or 'a, b y c' — and 'e' before an i-sound, as spoken."""
    if len(items) == 1:
        return items[0]
    head = fold(str(items[-1])).lower()
    y = "e" if head.startswith(("i", "hi")) and not head.startswith("hie") else "y"
    return ", ".join(items[:-1]) + f" {y} {items[-1]}"


def _es_listing(items: list) -> str:
    """At most three names, then a count of the rest."""
    if len(items) <= 3:
        return _es_join(items)
    remaining = len(items) - 3
    return (", ".join(items[:3])
            + f" y {_ES_NUMBER_WORDS.get(remaining, str(remaining))} más")


def _es_greeting(hour: int) -> str:
    # Not the English boundaries: "buenas tardes" runs until dusk, and
    # "buenas noches" at five in the afternoon is simply wrong.
    if hour < 12:
        return "Buenos días, señor."
    if hour < 19:
        return "Buenas tardes, señor."
    return "Buenas noches, señor."


def _es_clock(when: datetime) -> str:
    """'las 10 de la mañana', 'la 1:30 de la tarde' — the clock as it is said."""
    hour12 = when.hour % 12 or 12
    article = "la" if hour12 == 1 else "las"
    minutes = f":{when.minute:02d}" if when.minute else ""
    if when.hour == 0 or when.hour >= 19:
        part = "de la noche"
    elif when.hour < 12:
        part = "de la mañana"
    elif when.hour == 12:
        part = "del día"
    else:
        part = "de la tarde"
    return f"{article} {hour12}{minutes} {part}"


def _es_reset_time(when: datetime, now: datetime) -> str:
    """A reset time that names the day when it is not today — the same four
    shapes as `server._fmt_reset`, each one reading correctly after "hasta"."""
    clock = _es_clock(when)
    days = (when.date() - now.date()).days
    if days <= 0:
        return clock
    if days == 1:
        return f"mañana a {clock}"
    weekday = _ES_WEEKDAYS[when.weekday()]
    if days < 7:
        return f"el {weekday} a {clock}"
    return f"el {weekday} {when.day} de {_ES_MONTHS[when.month - 1]} a {clock}"


_ES_MX_DIRECTIVE = (
    " Speak Mexican Spanish. Everything you write is read aloud by a "
    "Spanish-language voice, so every reply is in natural Mexican Spanish — "
    "never English, whatever language a tool result, a file or your own "
    "instructions are written in. Address the user as \"señor\" wherever "
    "your instructions say \"sir\", and with usted, never tú. The stock "
    "phrases in your instructions are examples of tone, not words to say; "
    "say their Spanish: \"Me temo que no tengo ese dato, señor.\", \"Eso "
    "está fuera de mi alcance por ahora, señor.\", \"Enviado a chitauri, "
    "señor.\", \"Buscando, señor.\", \"¿Opus o Sonnet, señor?\". Tool "
    "results arrive in English: report what they say in Spanish, and never "
    "read an English sentence out. Names stay as they are — projects, "
    "sessions, files, commands, models, products — and go back to a tool "
    "exactly as the tool gave them to you. A message you send into a "
    "session, and the brief you write for a run or a build, is written in "
    "Spanish, as he said it. The words that clear your context are "
    "\"empieza de cero\": that is what you tell him to say, never \"start "
    "fresh\". The microphone now hears Spanish, so English names arrive "
    "bent: \"clod code\" or \"clau code\" is Claude Code, \"yarvis\" or "
    "\"jarbis\" is JARVIS, \"sonet\" is Sonnet.")

ES_MX = Language(
    code="es-MX",
    recognition="es-MX",
    # "Jarvis" on Fish Audio, a Spanish model of the dub's voice. Used only
    # when FISH_VOICE_ID is unset — a voice the user chose always wins.
    voice_id="ae74d1059517440aa6e5d2d50598d179",
    brain_directive=_ES_MX_DIRECTIVE,
    lines=_ES_MX_LINES,
    # Matched against the WHOLE utterance, never a word inside one, so "para"
    # (also "for") and "no" are safe here for the reason "no" is in English.
    cancel_words=("espera", "espérate", "no", "alto", "para", "párale",
                  "detente", "cancela", "cancelar", "un momento"),
    replay_filler=frozenset({"perdón", "disculpa", "disculpe", "jarvis",
                             "por", "favor", "eh", "este"}),
    replay_triggers=frozenset({
        ("repite",), ("repítelo",), ("repítalo",), ("repita",),
        ("repite", "eso"), ("repita", "eso"),
        ("otra", "vez"), ("dilo", "otra", "vez"), ("dígalo", "otra", "vez"),
        ("dilo", "de", "nuevo"), ("dígalo", "de", "nuevo"),
        ("qué", "dijiste"), ("qué", "dijo"), ("cómo", "dijiste"),
        ("cómo", "dijo"), ("cómo",), ("mande",),
    }),
    fresh_start_phrases=(
        "empieza de cero", "empecemos de cero", "empezar de cero",
        "empieza de nuevo", "empecemos de nuevo", "conversación nueva",
        "nueva conversación", "borra tu contexto", "limpia tu contexto",
        "despeja tu mente", "olvida esta conversación",
        "borrón y cuenta nueva",
    ),
    # No "no", unlike English: "No. 5" is rare here and "Me temo que no." is
    # the end of a sentence every other turn.
    abbreviations=frozenset({
        "sr", "sra", "srta", "dr", "dra", "lic", "ing", "prof", "ud", "uds",
        "etc", "aprox", "núm", "pág", "vs", "p.ej", "a.m", "p.m", "ee.uu",
    }),
    number_words=_ES_NUMBER_WORDS,
    # The English names a Spanish recogniser bends. Guesses from how the
    # sounds map, not yet measured live the way the English table was — add
    # to it from the "User:" lines in the server log.
    stt_corrections={
        r"\bcloud code\b": "Claude Code",
        r"\bclod code\b": "Claude Code",
        r"\bclau code\b": "Claude Code",
        r"\bclaud code\b": "Claude Code",
        r"\byarvis\b": "JARVIS",
        r"\byarbis\b": "JARVIS",
        r"\bjarbis\b": "JARVIS",
        r"\bllarvis\b": "JARVIS",
        r"\btravis\b": "JARVIS",
    },
    greeting=_es_greeting,
    listing=_es_listing,
    reset_time=_es_reset_time,
)

EN = Language(code="en", recognition="en-US")

LANGUAGES = {lang.code: lang for lang in (EN, ES_MX)}

# What an unset (or unrecognised) JARVIS_LANGUAGE means. Upstream this is
# English; this fork speaks Mexican Spanish out of the box, and
# `JARVIS_LANGUAGE=en` puts the English back.
DEFAULT = ES_MX

# What a person might reasonably type. "es" is Mexican Spanish because that is
# the only Spanish there is here; a second variety would have to take it back.
_ALIASES = {
    "en": EN, "en-us": EN, "en-gb": EN, "english": EN,
    "es": ES_MX, "es-mx": ES_MX, "es-419": ES_MX, "spanish": ES_MX,
    "espanol": ES_MX,
}


def resolve(value: Optional[str]) -> Optional[Language]:
    """The language `value` names, or None if it names none.

    The answer is always one of this file's own objects. `JARVIS_LANGUAGE` is
    a setting an HTTP route can write, and what it selects reaches the brain's
    system prompt — so the value chooses among sentences written here and is
    never itself printed anywhere.
    """
    key = fold(str(value or "")).strip().lower().replace("_", "-")
    return _ALIASES.get(key)


_warned_about: set = set()


def current() -> Language:
    """The language JARVIS is running in. Read from the environment on every
    call, like `data_paths.data_dir()`, so a test can set it per test."""
    raw = os.getenv(ENV_VAR, "")
    if not raw.strip():
        return DEFAULT
    lang = resolve(raw)
    if lang is None:
        if raw not in _warned_about:
            _warned_about.add(raw)
            log.warning("%s=%r is not a language JARVIS speaks (%s); using %s",
                        ENV_VAR, raw[:40], ", ".join(LANGUAGES), DEFAULT.code)
        return DEFAULT
    return lang


def t(english: str, **values) -> str:
    """`english`, in the language JARVIS is speaking, with `values` filled in.

    The English line is the key, so the sentence stays readable where it is
    spoken and a missing translation is English rather than a KeyError in the
    middle of an announcement. `values` are formatted into the template and
    never interpreted: a `{` in a project name is a `{` in the sentence.
    """
    template = current().lines.get(english, english)
    return template.format(**values) if values else template
