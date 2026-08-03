use std::ffi::OsString;
use std::path::PathBuf;
use std::str::FromStr;

use lamquant_lma_training_legacy::{
    install_termination_signal_forwarding, launch, LegacyEnvironment, LegacyTrainer,
};

fn usage() -> &'static str {
    "usage: lamquant-lma-training-legacy --git ABSOLUTE_GIT --checkout PATH \
     --trainer NAME --python ABSOLUTE_PYTHON --workspace ABSOLUTE_PATH \
     [--env NAME=VALUE]... -- LEGACY_TRAINER_ARGS...\n\
     lamquant-lma-training-legacy --list-trainers"
}

enum Action {
    Help,
    ListTrainers,
    Launch {
        git: OsString,
        checkout: PathBuf,
        trainer: LegacyTrainer,
        python: OsString,
        workspace: PathBuf,
        environment: Vec<LegacyEnvironment>,
        legacy_args: Vec<OsString>,
    },
}

fn set_once<T>(slot: &mut Option<T>, option: &str, value: T) -> Result<(), String> {
    if slot.is_some() {
        return Err(format!("duplicate {option}"));
    }
    *slot = Some(value);
    Ok(())
}

fn parse_args() -> Result<Action, String> {
    let mut git = None;
    let mut checkout = None;
    let mut trainer = None;
    let mut python = None;
    let mut workspace = None;
    let mut environment = Vec::new();
    let mut legacy_args = Vec::new();
    let mut args = std::env::args_os().skip(1);

    while let Some(argument) = args.next() {
        if argument == "--" {
            legacy_args.extend(args);
            break;
        }
        match argument.to_str() {
            Some("--help" | "-h") => return Ok(Action::Help),
            Some("--list-trainers") => return Ok(Action::ListTrainers),
            Some("--git") => {
                set_once(
                    &mut git,
                    "--git",
                    args.next()
                        .ok_or_else(|| "--git requires ABSOLUTE_GIT".to_owned())?,
                )?;
            }
            Some("--checkout") => {
                set_once(
                    &mut checkout,
                    "--checkout",
                    PathBuf::from(
                        args.next()
                            .ok_or_else(|| "--checkout requires PATH".to_owned())?,
                    ),
                )?;
            }
            Some("--trainer") => {
                let value = args
                    .next()
                    .ok_or_else(|| "--trainer requires NAME".to_owned())?;
                let value = value
                    .to_str()
                    .ok_or_else(|| "--trainer NAME must be UTF-8".to_owned())?;
                set_once(
                    &mut trainer,
                    "--trainer",
                    LegacyTrainer::from_str(value).map_err(|error| error.to_string())?,
                )?;
            }
            Some("--python") => {
                set_once(
                    &mut python,
                    "--python",
                    args.next()
                        .ok_or_else(|| "--python requires ABSOLUTE_PYTHON".to_owned())?,
                )?;
            }
            Some("--workspace") => {
                set_once(
                    &mut workspace,
                    "--workspace",
                    PathBuf::from(
                        args.next()
                            .ok_or_else(|| "--workspace requires ABSOLUTE_PATH".to_owned())?,
                    ),
                )?;
            }
            Some("--env") => {
                let assignment = args
                    .next()
                    .ok_or_else(|| "--env requires NAME=VALUE".to_owned())?;
                let assignment = assignment
                    .to_str()
                    .ok_or_else(|| "--env NAME=VALUE must be UTF-8".to_owned())?;
                let (name, value) = assignment
                    .split_once('=')
                    .ok_or_else(|| "--env requires NAME=VALUE".to_owned())?;
                environment.push(LegacyEnvironment::new(name, value)?);
            }
            _ => return Err(format!("unknown launcher argument {:?}", argument)),
        }
    }

    Ok(Action::Launch {
        git: git.ok_or_else(|| "--git is required".to_owned())?,
        checkout: checkout.ok_or_else(|| "--checkout is required".to_owned())?,
        trainer: trainer.ok_or_else(|| "--trainer is required".to_owned())?,
        python: python.ok_or_else(|| "--python is required".to_owned())?,
        workspace: workspace.ok_or_else(|| "--workspace is required".to_owned())?,
        environment,
        legacy_args,
    })
}

fn main() {
    let action = match parse_args() {
        Ok(action) => action,
        Err(error) => {
            eprintln!("{error}\n{}", usage());
            std::process::exit(2);
        }
    };
    let (git, checkout, trainer, python, workspace, environment, legacy_args) = match action {
        Action::Help => {
            println!("{}", usage());
            return;
        }
        Action::ListTrainers => {
            for trainer in LegacyTrainer::ALL {
                println!("{trainer}");
            }
            return;
        }
        Action::Launch {
            git,
            checkout,
            trainer,
            python,
            workspace,
            environment,
            legacy_args,
        } => (
            git,
            checkout,
            trainer,
            python,
            workspace,
            environment,
            legacy_args,
        ),
    };
    if let Err(error) = install_termination_signal_forwarding() {
        eprintln!("install termination signal forwarding: {error}");
        std::process::exit(1);
    }
    match launch(
        git,
        checkout,
        trainer,
        python,
        workspace,
        &environment,
        &legacy_args,
    ) {
        Ok(status) if status.success() => {}
        Ok(status) => std::process::exit(status.code().unwrap_or(1)),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(error.exit_code());
        }
    }
}
