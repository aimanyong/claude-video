"""Edit plan for the executive boardroom cut.

Every time below is in SOURCE seconds (the original recording). The final
timeline adds INTRO seconds of title card in front and OUTRO seconds of end
card after. On-screen text only restates what is spoken in the recording.
"""
INTRO = 3.0          # title card length before the recording starts
OUTRO = 4.0          # end card hold after the recording ends
SRC_DUR = 163.65     # length of the original recording
FPS = 30
W, H = 1920, 1080

PRESENTER = "Muhammad Aiman Bin Ali Azizan"
PRESENTER_ROLE = "LEASING TEAM  ·  PNB MERDEKA VENTURES"

TITLE_KICKER = "EXECUTIVE BRIEFING"
TITLE_MAIN = "Strategic Location Proposal"
TITLE_SUB = "OFFICE LEASING  ·  KUALA LUMPUR"

# (start, end, label, main line, optional sub line)
CALLOUTS = [
    (9.0, 18.4, "THE REQUIREMENT", "Office space for 400 people", "Malaysia or Thailand"),
    (31.3, 38.2, "GLOBAL TECHNOLOGY INVESTMENT", "Microsoft  ·  Google  ·  Amazon", "Investing billions in Malaysia"),
    (40.9, 52.3, "MULTILINGUAL WORKFORCE", "English  ·  Malay  ·  Mandarin", "Connecting head office and customers across Asia"),
    (53.1, 60.3, "GOVERNMENT FACILITATION", "Support through to work permit approval", None),
    (64.0, 72.7, "COST ADVANTAGE", "Office space costs less than other major Asian cities", "Lower cost of living"),
    (75.6, 81.3, "LANDMARK DISTINCTION", "Second tallest building in the world", "LEED Platinum green certification"),
    (82.7, 90.9, "ANCHOR TENANT", "Maybank headquarters", None),
    (97.4, 110.2, "WORKPLACE AMENITY", "Direct connection to 118 Mall", "Wide variety of food & beverage"),
    (113.7, 121.3, "CONNECTIVITY", "Easy access to MRT public transport", None),
    (132.6, 146.0, "SCALABILITY", "Additional space as your team expands", "Offered at an affordable price"),
]

# Agenda: the three questions he poses (start, end, [(reveal time, text)])
AGENDA = (18.9, 24.3, [(21.0, "Why Malaysia?"), (21.75, "Why Kuala Lumpur?"), (22.6, "Why this building?")])

# Chapter reveals: (start, number, title)
CHAPTERS = [
    (24.6, "01", "WHY MALAYSIA"),
    (60.7, "02", "WHY KUALA LUMPUR"),
    (73.0, "03", "WHY THIS BUILDING"),
    (122.0, "04", "ROOM TO GROW"),
]
CHAPTER_HOLD = 3.0

# Closing super over the final line (start, end)
CLOSING = (154.9, 163.4, "MERDEKA 118", "Let’s start your journey")

END_MAIN = "Merdeka 118"
END_SUB = "Thank you for your time."

# Camera: piecewise zoom targets (start, zoom multiplier). Smoothed with a
# gaussian so moves ease in and out; >1 pushes in on key statements, 1.0 is
# the breathing-room framing used around section changes.
ZOOM = [
    (0.0, 1.00), (2.5, 1.05), (8.8, 1.08), (18.6, 1.00),
    (24.5, 1.02), (31.0, 1.13), (38.6, 1.04), (40.8, 1.08), (53.0, 1.11),
    (60.5, 1.00), (64.0, 1.12),
    (72.9, 1.00), (75.2, 1.15), (82.5, 1.10), (91.0, 1.04), (97.2, 1.10),
    (111.0, 1.13), (121.8, 1.00), (125.5, 1.07), (132.5, 1.12),
    (146.2, 1.05), (151.0, 1.10), (155.0, 1.16),
]
ZOOM_BASE = 1.12      # base crop of the original frame
ZOOM_SIGMA = 0.85     # seconds; softness of camera moves

# Sound design cue points (output-timeline seconds are derived from these)
SFX_CHAPTER = [c[0] for c in CHAPTERS]
