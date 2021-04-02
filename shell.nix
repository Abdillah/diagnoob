{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python37Packages.pip
    python37Packages.tensorflow-build_2
    python37Packages.tensorflow-tensorboard_2
    python37Packages.Keras
    #python37Packages.flask

    # keep this line if you use bash
    bashInteractive
  ];
}
