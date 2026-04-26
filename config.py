POPULATION          = 100
GENERATIONS         = 50
MAX_STEPS           = 100_000
WORLD_WIDTH         = 500.0
WORLD_HEIGHT        = 430.0
PIPE_GAP            = 180.0
PIPE_SPEED          = 3.0
PIPE_SPACING        = 220.0
# 1.2 was found empirically: 0.5 fragmented the population into 40+ species by
# generation 3 and stalled diversity, while 3.0 collapsed everything into one
# species and killed structural innovation before it could compound. 1.2 keeps
# the species count in the single digits throughout training.
COMPAT_THRESHOLD    = 1.2
NUM_INPUTS          = 5  # bird_y, velocity, dx_pipe, gap_error, gap_size
REPLAY_TOP_K        = 20
REPLAY_CONVERGE_K   = 20  # stop training once this many top birds survive the window
REPLAY_STEPS        = 10_000  # max frames recorded per genome (~130 pipes)
