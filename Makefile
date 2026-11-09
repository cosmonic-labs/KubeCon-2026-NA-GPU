.PHONY: record retime play

# Runs every demo from its README and records it to docs/walkthrough.cast.
record:
	bash scripts/record.sh

# Re-times docs/walkthrough.raw.cast for playback, without recording again.
retime:
	node scripts/retime.mjs docs/walkthrough.raw.cast docs/walkthrough.cast

play:
	asciinema play docs/walkthrough.cast
