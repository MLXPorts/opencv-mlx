## OpenCV-MLX: NumPy-Free, MLX-Native Computer Vision

> **This is a specialized fork of OpenCV with all NumPy dependencies replaced by MLX.**
>
> **Key Changes:**
> - **MLX-Native:** All 223 Python files migrated from `numpy` to `mlx.core` (`np.` → `mx.`)
> - **NumPy-Free:** Complete removal of NumPy dependency
> - **Python 3.14+ Required:** Built for Python 3.14 free-threading
> - **Part of MLX Ecosystem:** Uses [mlx-precise](https://github.com/SolaceHarmony/mlx-precise) for tensor operations
> - **Apple Silicon Only:** Optimized for M1/M2/M3/M4 with Metal acceleration

**Requirements:**
- **Python 3.14 or later** (required for free-threading)
- **Apple Silicon Mac** (M1/M2/M3/M4)
- **macOS 11.0+**

**Installation:**
```bash
# Requires Python 3.14 free-threading build
python --version  # Should show Python 3.14.0 or later

# Install from git (Note: Build from source - see OpenCV docs)
git clone https://github.com/SolaceHarmony/opencv-mlx
cd opencv-mlx

# Will automatically install mlx-precise dependency
# See modules/python/package/setup.py for dependency configuration
```

**Migration Details:**
- 223 files modified (1818 insertions, 1817 deletions)
- Pattern: `import numpy as np` → `import mlx.core as mx`
- All `np.*` array operations replaced with `mx.*` equivalents

**Upstream:** Based on [OpenCV 4.x](https://github.com/opencv/opencv/tree/4.x)

---

## OpenCV: Open Source Computer Vision Library


### Resources

* Homepage: <https://opencv.org>
  * Courses: <https://opencv.org/courses>
* Docs: <https://docs.opencv.org/4.x/>
* Q&A forum: <https://forum.opencv.org>
  * previous forum (read only): <http://answers.opencv.org>
* Issue tracking: <https://github.com/opencv/opencv/issues>
* Additional OpenCV functionality: <https://github.com/opencv/opencv_contrib>
* Donate to OpenCV: <https://opencv.org/support/>


### Contributing

Please read the [contribution guidelines](https://github.com/opencv/opencv/wiki/How_to_contribute) before starting work on a pull request.

#### Summary of the guidelines:

* One pull request per issue;
* Choose the right base branch;
* Include tests and documentation;
* Clean up "oops" commits before submitting;
* Follow the [coding style guide](https://github.com/opencv/opencv/wiki/Coding_Style_Guide).

### Additional Resources

* [Submit your OpenCV-based project](https://form.jotform.com/233105358823151) for inclusion in Community Friday on opencv.org
* [Subscribe to the OpenCV YouTube Channel](http://youtube.com/@opencvofficial) featuring OpenCV Live, an hour-long streaming show
* [Follow OpenCV on LinkedIn](http://linkedin.com/company/opencv/) for daily posts showing the state-of-the-art in computer vision & AI
* [Apply to be an OpenCV Volunteer](https://form.jotform.com/232745316792159) to help organize events and online campaigns as well as amplify them
* [Follow OpenCV on Mastodon](http://mastodon.social/@opencv) in the Fediverse
* [Follow OpenCV on Twitter](https://twitter.com/opencvlive)
* [OpenCV.ai](https://opencv.ai): Computer Vision and AI development services from the OpenCV team.
