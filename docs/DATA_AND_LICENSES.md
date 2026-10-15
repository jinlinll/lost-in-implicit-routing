# Data and Licenses

Due to licensing restrictions on some third-party skills, we provide source lists rather than skill contents or download scripts. Readers can obtain skills from the linked repositories under their original licenses. External repositories and files may have changed since our experiments.

## Skill Sources

| File | Contents |
|---|---|
| [Source links](SKILL_SOURCES.md) | 670 recorded GitHub repositories |
| [Per-skill index](../manifests/skill_sources.jsonl) | Source links and definition hashes |
| [Unresolved sources](../manifests/unresolved_sources.json) | 474 entries without recorded sources |
| [Skill pool](../manifests/skill_pool.json) | 21,262 directory identifiers, hashes, and sampling order |

Source records identify repositories, which may include aggregators; exact skill paths and Git commits are not recorded. The index marks known restrictive notices as `restricted_local_notice` and other entries as `unreviewed`. Consult the original license when obtaining or using a skill.

## Benchmark

Obtain tasks from [SkillsBench](https://github.com/benchflow-ai/skillsbench). [The benchmark manifest](../manifests/benchmark.json) records the version and task list; [the patch](../manifests/benchmark_environment.patch) provides our environment and verifier changes. SkillsBench uses Apache-2.0; third-party materials within tasks retain their own terms.

## Code

The authors' original code and software documentation use the [MIT License](../LICENSE). The paper and its figures use [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Third-party materials retain their original licenses; see [NOTICE](../NOTICE.md).
