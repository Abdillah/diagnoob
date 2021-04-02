{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = pkgs; [
    python37Packages.pip
    python37Packages.tensorflow
    python37Packages.Keras
    python37Packages.flask

    # keep this line if you use bash
    bashInteractive
  ];
}
