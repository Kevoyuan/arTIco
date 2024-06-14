import inspect
import sys
from logging import Logger
from pathlib import Path
from typing import Union

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
import src.utils._custom_log as custom_log
from src.build._BasePipe import BasePipe
from src.tuner._BaseHyperparameterGenerator import BaseHyperparameterGenerator


class ObjectChecker:
    def __init__(self, log: Union[Logger, None] = None) -> None:
        """Checks objects for their compliance with a reference

        Args:
            log (Union[Logger, None], optional): logger. Defaults to None.
        """
        # logger
        self.__log: Logger = custom_log.init_logger(log_lvl=10) if log is None else log

    def pipeline(self, pipe: BasePipe) -> BasePipe:
        """Check pipeline for framework compatibility

        Args:
            pipe (BasePipe): pipeline

        Raises:
            AttributeError: raised if incompatibility found

        Returns:
            BasePipe: unchanged pipeline
        """
        self.__log.info("Check User Pipeline '%s' for Framework compatibility", pipe.__class__.__name__)
        # test init base pipe
        ref_pipe = BasePipe(work_dir=Path(), log=Logger)

        # check
        checks_passed = self.__check_class(ref_class=ref_pipe, check_class=pipe)

        # summary
        if checks_passed:
            self.__log.info("All checks for User Pipeline '%s' passed", pipe.__class__.__name__)
        else:
            self.__log.critical("User Pipeline '%s' not compatible with framework", pipe.__class__.__name__)
            raise AttributeError

        return pipe

    def hyperparameter_generator(self, generator: BaseHyperparameterGenerator) -> BaseHyperparameterGenerator:
        """Check hyperparameter generator for framework compatibility

        Args:
            generator (BaseHyperparameterGenerator): hyperparameter generator

        Raises:
            AttributeError: raised if incompatibility found

        Returns:
            BaseHyperparameterGenerator: unchanged hyperparameter generator
        """
        self.__log.info("Check Hyperparameter Generator '%s' for Framework compatibility", generator.__class__.__name__)
        # test init base pipe
        ref_generator = BaseHyperparameterGenerator(log=Logger)

        # check
        checks_passed = self.__check_class(ref_class=ref_generator, check_class=generator)

        # summary
        if checks_passed:
            self.__log.info("All checks for Hyperparameter Generator '%s' passed", generator.__class__.__name__)
        else:
            self.__log.critical("Hyperparameter Generator '%s' not compatible with framework", generator.__class__.__name__)
            raise AttributeError

        return generator

    def __check_class(self, ref_class: object, check_class: object) -> bool:
        """Check if class complies with reference

        Args:
            ref_class (object): initialized class, used as reference
            check_class (object): initialized class to compare with reference

        Returns:
            bool: True if check passed
        """
        # check __init__
        init_ok = self.__check_passed_init(ref_class=ref_class, check_class=check_class)

        # check methods
        methods_ok = self.__check_passed_methods(ref_class=ref_class, check_class=check_class)

        # check arguments of callable attributes
        arguments_ok = self.__check_passed_callable_arguments(ref_class=ref_class, check_class=check_class)

        return init_ok and methods_ok and arguments_ok

    def __check_passed_init(self, ref_class: object, check_class: object) -> bool:
        """Checks if __init__ signature is sufficient

        Args:
            ref_class (object): initialized class, used as reference
            check_class (object): initialized class to compare with reference

        Returns:
            bool: True if check passed
        """
        ref_parameters = set(inspect.signature(ref_class.__class__).parameters)
        check_parameters = set(inspect.signature(check_class.__class__).parameters)
        missing_parameters = ref_parameters - check_parameters
        additional_parameters = check_parameters - ref_parameters
        if missing_parameters:
            self.__log.critical("Init misses arguments: %s", missing_parameters)
        else:
            self.__log.info("Init OK")
        if additional_parameters:
            self.__log.warning("Init has additional arguments: %s - Potential undefined behavior", additional_parameters)

        return not missing_parameters

    def __check_passed_methods(self, ref_class: object, check_class: object) -> bool:
        """Checks if all required methods exist

        Args:
            ref_class (object): initialized class, used as reference
            check_class (object): initialized class to compare with reference

        Returns:
            bool: True if check passed
        """
        ref_attr = set(dir(ref_class))
        check_attr = set(dir(check_class))
        missing_attr = ref_attr - check_attr
        additional_attr = check_attr - ref_attr
        if missing_attr:
            self.__log.critical("Missing Attribute(s): %s", missing_attr)
        else:
            self.__log.info("Attributes OK")
        if additional_attr:
            self.__log.info("Additional attributes: %s", additional_attr)

        return not missing_attr

    def __check_passed_callable_arguments(self, ref_class: object, check_class: object) -> bool:
        """Checks if all methods have the required arguments

        Args:
            ref_class (object): initialized class, used as reference
            check_class (object): initialized class to compare with reference

        Returns:
            bool: True if check passed
        """
        # get missing attributes
        ref_attr = set(dir(ref_class))
        check_attr = set(dir(check_class))
        missing_attr = ref_attr - check_attr

        # check
        missing_inners = set([])
        for attr in dir(ref_class):
            if (
                not (attr.startswith("__") and attr.endswith("__"))
                and attr not in missing_attr
                and callable(ref_class.__getattribute__(attr))
                and callable(check_class.__getattribute__(attr))
            ):
                ref_inner_parameters = set(inspect.signature(ref_class.__getattribute__(attr)).parameters)
                check_inner_parameters = set(inspect.signature(check_class.__getattribute__(attr)).parameters)
                missing_inner_parameters = ref_inner_parameters - check_inner_parameters
                missing_inners |= missing_inner_parameters
                additional_inner_parameters = check_inner_parameters - ref_inner_parameters
                if missing_inner_parameters:
                    self.__log.critical("Method '%s' misses arguments: %s", attr, missing_inner_parameters)
                else:
                    self.__log.info("Method '%s' OK", attr)
                if additional_inner_parameters:
                    self.__log.info("Method '%s' has additional arguments: %s", attr, additional_inner_parameters)

        return not missing_inners
