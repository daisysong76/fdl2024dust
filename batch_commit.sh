#!/bin/bash

batch_size=10
counter=0
for file in $(git status -s | awk '{print $2}'); do
  git add "$file"
  counter=$((counter + 1))
  if [ $counter -ge $batch_size ]; then
    git commit -m "Batch commit"
    git push origin main  # or `master`
    counter=0
  fi
done

# Commit remaining files
if [ $counter -gt 0 ]; then
  git commit -m "Final batch commit"
  git push origin main  # or `master`
fi
